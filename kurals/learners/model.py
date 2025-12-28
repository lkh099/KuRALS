"""Class to train a PyTorch model"""
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from functools import partial
import random
import time

from kurals.loaders.dataloaders import KuRALSDataset
from kurals.learners.tester import Tester
from kurals.utils.functions import normalize, define_loss, get_transformations
from kurals.utils.tensorboard_visualizer import TensorboardMultiLossVisualizer
from kurals.utils.distributed_utils import get_rank, reduce_value


class Model(nn.Module):
    """Class to train a model

    PARAMETERS
    ----------
    net: PyTorch Model
        Network to train
    data: dict
        Parameters and configurations for training
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
        self.annot_type = self.cfg['annot_type']
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
        self.viz_step = self.cfg['viz_step']
        self.torch_seed = self.cfg['torch_seed']
        self.numpy_seed = self.cfg['numpy_seed']
        self.nb_classes = self.cfg['nb_classes']
        self.custom_loss = self.cfg['custom_loss']
        self.comments = self.cfg['comments']
        self.n_frames = self.cfg['nb_input_channels']
        self.transform_names = self.cfg['transformations'].split(',')
        self.norm_type = self.cfg['norm_type']
        self.is_shuffled = self.cfg['shuffle']
        self.device = self.cfg['device']
        self.distributed = self.cfg['distributed']
        self.num_workers = self.cfg['num_workers']
        self.rank = get_rank()
        if self.rank == 0:
            self.writer = SummaryWriter(self.paths['writer'])
            self.visualizer = TensorboardMultiLossVisualizer(self.writer)
            self.tester = Tester(self.cfg, self.visualizer)
        else:
            self.tester = Tester(self.cfg)
        self.results = dict()

    def train(self, add_temp=False):
        """
        Method to train a network

        PARAMETERS
        ----------
        add_temp: boolean
            Add a temporal dimension during training?
            Considering the input as a sequence.
            Default: False
        """
        if self.rank == 0:
            self.writer.add_text('Comments', self.comments)
        train_loader, val_loader, test_loader = self.dataloaders
        transformations = get_transformations(self.transform_names,
                                              sizes=(self.w_size, self.h_size))
        rd_criterion = define_loss(self.dataset, 'range_doppler', self.custom_loss, self.device)
        nb_losses = len(rd_criterion)
        running_losses = list()
        rd_running_losses = list()
        rd_running_global_losses = [list(), list()]
        optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        # lt @20230417
        if self.schedular_type == 'exp':
            scheduler = ExponentialLR(optimizer, gamma=0.9)
        elif self.schedular_type == 'cos':
            scheduler = CosineAnnealingLR(optimizer, T_max=self.T_max-1, eta_min=1e-7)
        else:
            raise KeyError("we only implement schedular of exp and cos")
        
        iteration = 0
        best_val_prec = 0
        best_val_recall = 0
        # lt @20230522
        flag_save = False
        name_result = []
        best_val_dop = 0
        best_val_ang = 0
        best_val_dop_plus_ang = 0
        best_test_dop = 0
        best_test_ang = 0
        best_test_dop_plus_ang = 0
        # zlw@20211012
        # if torch.cuda.device_count() > 1:
        #     self.net = nn.DataParallel(self.net, device_ids=[2, 3])
            #self.net = nn.DataParallel(self.net, device_ids=[3])
        # self.net.to(self.device)

        for epoch in range(self.nb_epochs):
            # lt @20230417
            if self.schedular_type == 'exp':
                if epoch % self.lr_step == 0 and epoch != 0:
                    scheduler.step()
            else:
                if epoch != 0 and epoch < self.T_max:
                    scheduler.step()
            for _, sequence_data in enumerate(train_loader):
                seq_name, seq = sequence_data
                if self.dataset == 'KuRALS_CW':
                    path_to_frames = self.paths['KuRALS_CW'] / seq_name[0]
                elif self.dataset == 'KuRALS_PD':
                    path_to_frames = self.paths['KuRALS_PD'] / seq_name[0]
                else:
                    raise KeyError(f'Dataset {self.dataset} has not been supported yet.')
                
                kurals_dataset = KuRALSDataset(seq,
                                                self.annot_type,
                                                path_to_frames,
                                                self.process_signal,
                                                self.n_frames,
                                                transformations,
                                                add_temp)
                
                if self.distributed:
                    sampler_train = torch.utils.data.distributed.DistributedSampler(kurals_dataset)
                    sampler_train.set_epoch(epoch)
                else:
                    sampler_train = torch.utils.data.RandomSampler(kurals_dataset)
                train_batch_sampler = torch.utils.data.BatchSampler(sampler_train, self.batch_size, drop_last=True)
                frame_dataloader = DataLoader(kurals_dataset,
                                              batch_sampler=train_batch_sampler,
                                              num_workers=self.num_workers,
                                              worker_init_fn=partial(worker_init_fn, rank=get_rank(), seed=self.numpy_seed))
                for _, frame in enumerate(frame_dataloader):
                    rd_data = frame['rd_matrix'].to(self.device).float()
                    rd_mask = frame['rd_mask'].to(self.device).float()
                    rd_data = normalize(rd_data, self.dataset, 'range_doppler', norm_type=self.norm_type)
                    
                    optimizer.zero_grad()
                    
                    rd_outputs = self.net(rd_data)
                    rd_outputs = rd_outputs.to(self.device)

                    rd_losses = [c(rd_outputs, torch.argmax(rd_mask, axis=1))
                                for c in rd_criterion]
                    rd_loss = torch.mean(torch.stack(rd_losses))

                    loss = rd_loss
                    rd_losses_reduced = reduce_value(rd_losses)
                    rd_loss_reduced = reduce_value(rd_loss)
                    loss_reduced = reduce_value(loss)
                    
                    loss.backward()
                    optimizer.step()
            
                    if self.rank == 0:
                        running_losses.append(loss_reduced.data.cpu().numpy()[()])
                        rd_running_losses.append(rd_loss_reduced.data.cpu().numpy()[()])
                        rd_running_global_losses[0].append(rd_losses_reduced[0].data.cpu().numpy()[()])
                        rd_running_global_losses[1].append(rd_losses_reduced[1].data.cpu().numpy()[()])

                        if iteration % self.loss_step == 0:
                            train_loss = np.mean(running_losses)
                            rd_train_loss = np.mean(rd_running_losses)
                            rd_train_losses = [np.mean(sub_loss) for sub_loss in rd_running_global_losses]
                            # zlw@20220302
                            print('[{}][Epoch {}/{}, iter {}]: '
                                'Train loss {}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
                                                        epoch+1,
                                                        self.nb_epochs,
                                                        iteration,
                                                        train_loss))
                            print('[Epoch {}/{}, iter {}]: '
                                'Train losses: RD={}'.format(epoch+1,
                                                                    self.nb_epochs,
                                                                    iteration,
                                                                    rd_train_loss))
                            
                    
                            self.visualizer.update_multi_train_loss(train_loss, rd_train_loss,
                                                                        rd_train_losses, iteration)
                            running_losses = list()
                            rd_running_losses = list()
                            ra_running_losses = list()
                            # zlw@20220107 get_lr() --> get_last_lr()
                            self.visualizer.update_learning_rate(scheduler.get_last_lr()[0], iteration)

                    if iteration % self.val_step == 0 and iteration > 0:
                        if iteration % self.viz_step == 0 and iteration > 0:
                            val_metrics = self.tester.predict(self.net, val_loader, iteration,
                                                              add_temp=add_temp, mode='Validation', iteration_loss=iteration)
                        else:
                            val_metrics = self.tester.predict(self.net, val_loader, add_temp=add_temp, mode='Validation', iteration_loss=iteration)

                        if self.rank == 0:
                            self.visualizer.update_multi_val_metrics(val_metrics, iteration)
                            print('[Epoch {}/{}] Val losses: '
                                'RD={}'.format(epoch+1,
                                                        self.nb_epochs,
                                                        val_metrics['range_doppler']['loss'],
                                                        ))
                            
                            print('[Epoch {}/{}] Val Pixel Prec: '
                                'RD={}'.format(epoch+1,
                                                        self.nb_epochs,
                                                        val_metrics['range_doppler']['prec'],
                                                        ))
                            # zlw@20220227
                            print('[Epoch {}/{}] Val mIoU: '
                                'RD={}'.format(epoch+1,
                                                        self.nb_epochs,
                                                        val_metrics['range_doppler']['miou'],
                                                        ))
                            print('[Epoch {}/{}] Val Dice: '
                                'RD={}'.format(epoch+1,
                                                        self.nb_epochs,
                                                        val_metrics['range_doppler']['dice'],
                                                        ))

                            # lt @20230522
                            if val_metrics['range_doppler']['dice'] > best_val_dop and iteration > 0:
                                best_val_dop = val_metrics['range_doppler']['dice']
                                flag_save = True
                                name_result.append('val_doppler')
                            
                        # saving results w/ best dice zlw@20220704
                        test_metrics = self.tester.predict(self.net, test_loader,
                                                               add_temp=add_temp, mode='Test', iteration_loss=iteration)
                        if self.rank == 0:
                            self.visualizer.update_multi_test_metrics(test_metrics, iteration)
                            print('[Epoch {}/{}] Test losses: '
                                'RD={}'.format(epoch+1,
                                                        self.nb_epochs,
                                                        test_metrics['range_doppler']['loss']))
                            print('[Epoch {}/{}] Test Prec: '
                                'RD={}'.format(epoch+1,
                                                        self.nb_epochs,
                                                        test_metrics['range_doppler']['prec'],
                                                        ))
                            # zlw@20220227
                            print('[Epoch {}/{}] Test mIoU: '
                                'RD={}'.format(epoch+1,
                                                        self.nb_epochs,
                                                        test_metrics['range_doppler']['miou'],
                                                        ))
                            print('[Epoch {}/{}] Test Dice: '
                                'RD={}'.format(epoch+1,
                                                        self.nb_epochs,
                                                        test_metrics['range_doppler']['dice'],
                                                        ))

                            self.results['epoch'] = epoch + 1
                            self.results['rd_train_loss'] = rd_train_loss.item()
                            self.results['train_loss'] = train_loss.item()
                            self.results['val_metrics'] = val_metrics
                            self.results['test_metrics'] = test_metrics


                            if test_metrics['range_doppler']['dice'] > best_test_dop and iteration > 0:
                                best_test_dop = test_metrics['range_doppler']['dice']
                                flag_save = True
                                name_result.append('test_doppler')
                            
                            if flag_save:
                                self._save_results(name_result)
                                flag_save = False
                                name_result = []
                        self.net.train()  # Train mode after evaluation process
                    iteration += 1
        
        if self.rank == 0:
            self.writer.close()

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

    def _save_results(self, names):
        for name in names:
            results_path = self.paths['results'] / (name + '_' + 'results.json')
            model_path = self.paths['results'] / (name + '_' + 'model.pt')
            with open(results_path, "w") as fp:
                json.dump(self.results, fp)
            torch.save(self.net.module.state_dict(), model_path)

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