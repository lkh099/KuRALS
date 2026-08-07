"""Class to train KuRALSNetNPU (detection).

Kept separate from kurals.learners.model.Model on purpose: that class and its
Tester are built around per-pixel segmentation metrics (mIoU, Dice, pixel
precision/recall via a confusion matrix), which don't apply to a center-point
detector. This mirrors the same config-driven epoch/iteration loop, optimizer
and scheduler conventions, but logs detection losses only and evaluates with
validation loss rather than segmentation metrics.
"""
import time
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from functools import partial

from kurals.legacy.dataloaders_detect import KuRALSDetectionDataset
from kurals.legacy.detection_loss import DetectionLoss
from kurals.utils.functions import normalize, get_transformations
from kurals.utils.distributed_utils import get_rank, reduce_value


class DetectionModel(nn.Module):
    """Class to train KuRALSNetNPU

    PARAMETERS
    ----------
    net: PyTorch Model
        KuRALSNetNPU instance to train
    data: dict
        Parameters and configurations for training (same shape as produced
        by kurals.learners.initializer.Initializer, reused as-is since
        sequence/frame splitting is dataset-level, not task-level)
    """

    def __init__(self, net, data):
        super().__init__()
        self.net = net
        self.cfg = data['cfg']
        self.paths = data['paths']
        self.dataloaders = data['dataloaders']
        self.model_name = self.cfg['model']
        self.dataset = self.cfg['dataset']
        self.process_signal = self.cfg['process_signal']
        self.w_size = self.cfg['w_size']
        self.h_size = self.cfg['h_size']
        self.batch_size = self.cfg['batch_size']
        self.nb_epochs = self.cfg['nb_epochs']
        self.lr = self.cfg['lr']
        self.lr_step = self.cfg['lr_step']
        self.schedular_type = self.cfg['schedular']
        self.T_max = self.cfg['Tmax']
        self.loss_step = self.cfg['loss_step']
        self.val_step = self.cfg['val_step']
        self.torch_seed = self.cfg['torch_seed']
        self.numpy_seed = self.cfg['numpy_seed']
        self.nb_classes = self.cfg['nb_classes']
        self.comments = self.cfg['comments']
        self.n_frames = self.cfg['nb_input_channels']
        self.transform_names = self.cfg['transformations'].split(',')
        self.norm_type = self.cfg['norm_type']
        self.device = self.cfg['device']
        self.distributed = self.cfg['distributed']
        self.num_workers = self.cfg['num_workers']
        self.heatmap_weight = self.cfg.get('heatmap_weight', 1.0)
        self.offset_weight = self.cfg.get('offset_weight', 1.0)
        self.heatmap_pos_weight = self.cfg.get('heatmap_pos_weight', 1.0)
        self.gamma = self.cfg.get('gamma', 2.0)
        self.heatmap_loss_type = self.cfg.get('heatmap_loss_type', 'focal')
        self.weight_decay = self.cfg.get('weight_decay', 0.0)
        self.grad_clip_norm = self.cfg.get('grad_clip_norm', None)
        # QAT recipe: train in plain float for quant_warmup_iters (lets BN stats and
        # weights settle before quantization noise is introduced), then flip on the
        # NPU's exact int8 pipeline (see kurals/models/quant.py) for the rest of
        # training. quant_freeze_iters stops updating the activation-range observers
        # once calibration has stabilized, leaving only the weights to keep adapting.
        self.quant_warmup_iters = self.cfg.get('quant_warmup_iters', None)
        self.quant_freeze_iters = self.cfg.get('quant_freeze_iters', None)
        # LR warmup: previously nonexistent -- Adam started at the full peak LR from
        # iteration 0 (CosineAnnealingLR.step() is only ever called at epoch boundaries
        # after epoch 0, see train()'s scheduler.step() calls below, so all of epoch 0
        # ran at the un-annealed peak LR). Linearly ramps from
        # lr * lr_warmup_start_factor up to lr over the first lr_warmup_iters
        # iterations by overwriting optimizer.param_groups directly; once warmup ends
        # the scheduler's own epoch-boundary steps take over unchanged (it was never
        # touched during warmup, so its internal state/base_lrs are unaffected).
        self.lr_warmup_iters = self.cfg.get('lr_warmup_iters', 500)
        self.lr_warmup_start_factor = self.cfg.get('lr_warmup_start_factor', 0.1)
        # Rolling "last" checkpoint saved every checkpoint_step iterations, for crash
        # recovery and for resuming training with different hyperparameters across
        # repeated runs (see --resume in train_detect.py). Checkpoints only actually
        # land at epoch boundaries (see train()'s docstring note on resume semantics).
        self.checkpoint_step = self.cfg.get('checkpoint_step', 1000)
        self.rank = get_rank()
        if self.rank == 0:
            self.writer = SummaryWriter(self.paths['writer'])
        self.best_val_loss = float('inf')

    def train(self, resume_path=None):
        """Method to train KuRALSNetNPU

        PARAMETERS
        ----------
        resume_path: str or Path, optional
            Path to a checkpoint written by _save_checkpoint (i.e. a dict with
            'net', 'optimizer', 'scheduler', 'epoch', 'iteration', 'best_val_loss'
            -- not a bare state_dict). Resumes training from the epoch *after*
            the one during which the checkpoint was saved: this loop only ever
            checkpoints at epoch boundaries (a periodic mid-epoch "last" save
            exists purely for crash recovery, and still resumes at the next
            epoch boundary -- partial-epoch progress within it is redone, since
            the per-sequence/per-frame dataloaders are rebuilt fresh each epoch
            and don't support resuming mid-stream). iteration, the optimizer,
            the scheduler, and the quantization enabled/frozen state all resume
            exactly, so logging and the LR schedule stay continuous.
        """
        if self.rank == 0:
            self.writer.add_text('Comments', self.comments)
        train_loader, val_loader, _ = self.dataloaders
        transformations = get_transformations(self.transform_names,
                                              sizes=(self.w_size, self.h_size))
        criterion = DetectionLoss(heatmap_weight=self.heatmap_weight,
                                   offset_weight=self.offset_weight,
                                   pos_weight=self.heatmap_pos_weight,
                                   gamma=self.gamma,
                                   heatmap_loss_type=self.heatmap_loss_type)
        n_fg_classes = self.net.n_fg_classes

        optimizer = optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if self.schedular_type == 'exp':
            scheduler = ExponentialLR(optimizer, gamma=0.9)
        elif self.schedular_type == 'cos':
            scheduler = CosineAnnealingLR(optimizer, T_max=self.T_max - 1, eta_min=1e-7)
        else:
            raise KeyError("we only implement schedular of exp and cos")

        iteration = 0
        start_epoch = 0
        if resume_path is not None:
            iteration, start_epoch = self._load_checkpoint(resume_path, optimizer, scheduler)
            # Restore quantization state for thresholds already passed at the resume
            # point; thresholds still ahead are handled by the normal per-iteration
            # checks below, unchanged.
            if self.quant_warmup_iters is not None and iteration >= self.quant_warmup_iters:
                self.net.set_quant_enabled(True)
            if self.quant_freeze_iters is not None and iteration >= self.quant_freeze_iters:
                self.net.freeze_observers()
            if self.rank == 0:
                print(f'Resumed from {resume_path}: epoch {start_epoch}, iteration {iteration}, '
                      f'best_val_loss {self.best_val_loss}')

        running_losses, running_heatmap, running_offset = [], [], []

        for epoch in range(start_epoch, self.nb_epochs):
            if self.schedular_type == 'exp':
                if epoch % self.lr_step == 0 and epoch != 0:
                    scheduler.step()
            else:
                if epoch != 0 and epoch < self.T_max:
                    scheduler.step()

            for _, sequence_data in enumerate(train_loader):
                seq_name, seq = sequence_data
                if self.dataset in ('KuRALS_CW', 'KuRALS_PD'):
                    path_to_frames = self.paths[self.dataset] / seq_name[0]
                else:
                    raise KeyError(f'Dataset {self.dataset} has not been supported yet.')

                detect_dataset = KuRALSDetectionDataset(seq, path_to_frames,
                                                          self.process_signal,
                                                          self.n_frames,
                                                          n_fg_classes,
                                                          transformations)

                if self.distributed:
                    sampler_train = torch.utils.data.distributed.DistributedSampler(detect_dataset)
                    sampler_train.set_epoch(epoch)
                else:
                    sampler_train = torch.utils.data.RandomSampler(detect_dataset)
                train_batch_sampler = torch.utils.data.BatchSampler(sampler_train, self.batch_size, drop_last=True)
                frame_dataloader = DataLoader(detect_dataset,
                                              batch_sampler=train_batch_sampler,
                                              num_workers=self.num_workers,
                                              worker_init_fn=partial(worker_init_fn, rank=get_rank(), seed=self.numpy_seed))

                for _, frame in enumerate(frame_dataloader):
                    if self.quant_warmup_iters is not None and iteration == self.quant_warmup_iters:
                        self.net.set_quant_enabled(True)
                        if self.rank == 0:
                            print(f'[iter {iteration}] QAT enabled (float warm-up done)')
                    if self.quant_freeze_iters is not None and iteration == self.quant_freeze_iters:
                        self.net.freeze_observers()
                        if self.rank == 0:
                            print(f'[iter {iteration}] quantization observers frozen')

                    if iteration < self.lr_warmup_iters:
                        warmup_frac = iteration / max(self.lr_warmup_iters, 1)
                        warmup_lr = self.lr * (self.lr_warmup_start_factor +
                                                (1.0 - self.lr_warmup_start_factor) * warmup_frac)
                        for group in optimizer.param_groups:
                            group['lr'] = warmup_lr
                        if self.rank == 0 and iteration == 0:
                            print(f'[iter 0] LR warmup: {warmup_lr:.3g} -> {self.lr:.3g} '
                                  f'over {self.lr_warmup_iters} iters')

                    rd_data = frame['rd_matrix'].to(self.device).float()
                    rd_data = normalize(rd_data, self.dataset, 'range_doppler', norm_type=self.norm_type)
                    target = {
                        'heatmap': frame['heatmap'].to(self.device).float(),
                        'offset': frame['offset'].to(self.device).float(),
                        'mask': frame['mask'].to(self.device).float(),
                    }

                    optimizer.zero_grad()
                    outputs = self.net(rd_data).to(self.device)
                    loss, heatmap_loss, offset_loss = criterion(outputs, target)
                    loss.backward()
                    if self.grad_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip_norm)
                    optimizer.step()

                    loss_reduced = reduce_value(loss)
                    heatmap_loss_reduced = reduce_value(heatmap_loss)
                    offset_loss_reduced = reduce_value(offset_loss)

                    if self.rank == 0:
                        running_losses.append(loss_reduced.data.cpu().numpy()[()])
                        running_heatmap.append(heatmap_loss_reduced.data.cpu().numpy()[()])
                        running_offset.append(offset_loss_reduced.data.cpu().numpy()[()])

                        if iteration % self.loss_step == 0:
                            train_loss = np.mean(running_losses)
                            print('[{}][Epoch {}/{}, iter {}]: Train loss {} '
                                  '(heatmap={}, offset={})'.format(
                                      time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
                                      epoch + 1, self.nb_epochs, iteration, train_loss,
                                      np.mean(running_heatmap), np.mean(running_offset)))
                            self.writer.add_scalar('train/loss', train_loss, iteration)
                            self.writer.add_scalar('train/heatmap_loss', np.mean(running_heatmap), iteration)
                            self.writer.add_scalar('train/offset_loss', np.mean(running_offset), iteration)
                            self.writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], iteration)
                            running_losses, running_heatmap, running_offset = [], [], []

                    if iteration % self.val_step == 0 and iteration > 0:
                        val_loss = self._validate(val_loader, criterion, transformations, n_fg_classes)
                        if self.rank == 0:
                            print('[Epoch {}/{}] Val loss: {}'.format(epoch + 1, self.nb_epochs, val_loss))
                            self.writer.add_scalar('val/loss', val_loss, iteration)
                            if val_loss < self.best_val_loss:
                                self.best_val_loss = val_loss
                                self._save_checkpoint('best', optimizer, scheduler, epoch, iteration)
                        self.net.train()

                    if self.rank == 0 and iteration % self.checkpoint_step == 0 and iteration > 0:
                        self._save_checkpoint('last', optimizer, scheduler, epoch, iteration)

                    iteration += 1

            if self.rank == 0:
                self._save_checkpoint('last', optimizer, scheduler, epoch, iteration)

        if self.rank == 0:
            self.writer.close()

    def _validate(self, val_loader, criterion, transformations, n_fg_classes):
        self.net.eval()
        losses = []
        with torch.no_grad():
            for _, sequence_data in enumerate(val_loader):
                seq_name, seq = sequence_data
                path_to_frames = self.paths[self.dataset] / seq_name[0]
                detect_dataset = KuRALSDetectionDataset(seq, path_to_frames,
                                                          self.process_signal,
                                                          self.n_frames,
                                                          n_fg_classes,
                                                          transformations)
                frame_dataloader = DataLoader(detect_dataset, batch_size=self.batch_size,
                                              shuffle=False, num_workers=self.num_workers)
                for _, frame in enumerate(frame_dataloader):
                    rd_data = frame['rd_matrix'].to(self.device).float()
                    rd_data = normalize(rd_data, self.dataset, 'range_doppler', norm_type=self.norm_type)
                    target = {
                        'heatmap': frame['heatmap'].to(self.device).float(),
                        'offset': frame['offset'].to(self.device).float(),
                        'mask': frame['mask'].to(self.device).float(),
                    }
                    outputs = self.net(rd_data).to(self.device)
                    loss, _, _ = criterion(outputs, target)
                    losses.append(reduce_value(loss).data.cpu().numpy()[()])
        return float(np.mean(losses)) if losses else float('inf')

    def _save_checkpoint(self, name, optimizer=None, scheduler=None, epoch=None, iteration=None):
        """Save a full resumable checkpoint (not just weights): net state_dict
        (which includes the QAT observer buffers, so calibration survives a
        resume too), optimizer, scheduler, and loop position. 'best'/'last' are
        the two tags actually used by train(); any name works for one-off saves.
        """
        net = self.net.module if hasattr(self.net, 'module') else self.net
        checkpoint = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict() if optimizer is not None else None,
            'scheduler': scheduler.state_dict() if scheduler is not None else None,
            'epoch': epoch,
            'iteration': iteration,
            'best_val_loss': self.best_val_loss,
            'cfg': self.cfg,
        }
        torch.save(checkpoint, self.paths['results'] / f'{name}_checkpoint.pt')

    def _load_checkpoint(self, path, optimizer, scheduler):
        """Load a checkpoint written by _save_checkpoint and restore net,
        optimizer, scheduler, and self.best_val_loss in place.

        RETURNS
        -------
        (iteration, start_epoch): where to resume the loop (start_epoch = the
        checkpointed epoch + 1, since saves only happen at epoch boundaries --
        see train()'s docstring).
        """
        checkpoint = torch.load(path, map_location=self.device)
        net = self.net.module if hasattr(self.net, 'module') else self.net
        net.load_state_dict(checkpoint['net'])
        if checkpoint['optimizer'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if checkpoint['scheduler'] is not None:
            scheduler.load_state_dict(checkpoint['scheduler'])
        self.best_val_loss = checkpoint['best_val_loss']
        return checkpoint['iteration'], checkpoint['epoch'] + 1

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.)
        elif isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.uniform_(m.weight, 0., 1.)
            nn.init.constant_(m.bias, 0.)

    def _set_seeds(self):
        torch.cuda.manual_seed_all(self.torch_seed)
        torch.manual_seed(self.torch_seed)
        np.random.seed(self.numpy_seed)
        random.seed(self.numpy_seed)


def worker_init_fn(worker_id, rank, seed):
    worker_seed = rank + seed
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
