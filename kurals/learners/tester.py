"""Class to test a model"""
import json
import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from kurals.utils.functions import transform_masks_viz, get_metrics, normalize, define_loss, get_transformations, get_qualitatives
from kurals.utils.paths import Paths
from kurals.utils.metrics import Evaluator
from kurals.loaders.dataloaders import KuRALSDataset
from kurals.utils.distributed_utils import reduce_value, get_rank, all_gather
from kurals.utils.profiler import timings

class Tester:
    """
    Class to test a model

    PARAMETERS
    ----------
    cfg: dict
        Configuration parameters used for train/test
    visualizer: object or None
        Add a visulization during testing
        Default: None
    """

    def __init__(self, cfg, visualizer=None):
        self.cfg = cfg
        self.visualizer = visualizer
        self.model = self.cfg['model']
        self.dataset = self.cfg['dataset']
        self.nb_classes = self.cfg['nb_classes']
        self.annot_type = self.cfg['annot_type']
        self.process_signal = self.cfg['process_signal']
        self.w_size = self.cfg['w_size']
        self.h_size = self.cfg['h_size']
        self.n_frames = self.cfg['nb_input_channels']
        self.batch_size = self.cfg['batch_size']
        self.device = self.cfg['device']
        self.custom_loss = self.cfg['custom_loss']
        self.transform_names = self.cfg['transformations'].split(',')
        self.norm_type = self.cfg['norm_type']
        self.distributed = self.cfg['distributed']
        self.num_workers = self.cfg['num_workers']
        self.paths = Paths().get()
        self.test_results = dict()

    def predict(self, net, seq_loader, iteration=None, get_quali=False, add_temp=False, timer=False, mode='Test', iteration_loss=0):
        """
        Method to predict on a given dataset using a fixed model

        PARAMETERS
        ----------
        net: PyTorch Model
            Network to test
        seq_loader: DataLoader
            Specific to the dataset used for test
        iteration: int
            Iteration used to display visualization
            Default: None
        get_quali: boolean
            If you want to save qualitative results
            Default: False
        add_temp: boolean
            Is the data are considered as a sequence
            Default: False
        """
        net.eval()
        transformations = get_transformations(self.transform_names, split='test',
                                              sizes=(self.w_size, self.h_size))
        rd_criterion = define_loss(self.dataset, 'range_doppler', self.custom_loss, self.device)
        nb_losses = len(rd_criterion)
        running_losses = list()
        rd_running_losses = list()
        rd_running_global_losses = [list(), list()]
        rd_metrics = Evaluator(num_class=self.nb_classes)
        if iteration:
            rand_seq = np.random.randint(len(seq_loader))
        with torch.no_grad():
            for i, sequence_data in enumerate(seq_loader):
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
                    sampler_test = torch.utils.data.distributed.DistributedSampler(kurals_dataset, shuffle=False)
                else:
                    sampler_test = torch.utils.data.SequentialSampler(kurals_dataset)
                # lt@20240301 for pdr dataset, fix batch size to 2.
                if self.dataset == 'KuRALS_CW':
                    frame_dataloader = DataLoader(kurals_dataset,
                                              shuffle=False,
                                              sampler=sampler_test,
                                              batch_size=self.batch_size,
                                              num_workers=self.num_workers)
                elif self.dataset == 'KuRALS_PD':
                    frame_dataloader = DataLoader(kurals_dataset,
                                              shuffle=False,
                                              sampler=sampler_test,
                                              batch_size=1,
                                              num_workers=0)
                else:
                    raise KeyError(f'Dataset {self.dataset} has not been supported yet.')
                
                if iteration and i == rand_seq:
                    rand_frame = np.random.randint(len(frame_dataloader))
                if get_quali:
                    quali_iter_rd = self.n_frames-1
                for j, frame in enumerate(frame_dataloader):
                    rd_data = frame['rd_matrix'].to(self.device).float()
                    rd_mask = frame['rd_mask'].to(self.device).float()
                    rd_data = normalize(rd_data, self.dataset, 'range_doppler', norm_type=self.norm_type)
                    
                    rd_outputs = net(rd_data)
                    
                    rd_outputs = rd_outputs.to(self.device)

                    if get_quali:
                        quali_iter_rd = get_qualitatives(rd_outputs, rd_mask, self.paths,
                                                         seq_name, quali_iter_rd, 'range_doppler',
                                                         self.dataset)

                    rd_metrics.add_batch_dist(torch.argmax(rd_mask, axis=1),
                                         torch.argmax(rd_outputs, axis=1))

                    rd_losses = [c(rd_outputs, torch.argmax(rd_mask, axis=1))
                                    for c in rd_criterion]
                    rd_loss = torch.mean(torch.stack(rd_losses))

                    loss = rd_loss
                    rd_losses = reduce_value(rd_losses)
                    rd_loss = reduce_value(rd_loss)
                    loss = reduce_value(loss)
                    
                    running_losses.append(loss.data.cpu().numpy()[()])
                    rd_running_losses.append(rd_loss.data.cpu().numpy()[()])
                    rd_running_global_losses[0].append(rd_losses[0].data.cpu().numpy()[()])
                    rd_running_global_losses[1].append(rd_losses[1].data.cpu().numpy()[()])
                
                    if iteration and i == rand_seq:
                        if j == rand_frame:
                            rd_pred_masks = torch.argmax(rd_outputs, axis=1)[:5]
                            rd_gt_masks = torch.argmax(rd_mask, axis=1)[:5]
                            rd_pred_grid = make_grid(transform_masks_viz(rd_pred_masks,
                                                                         self.nb_classes))
                            rd_gt_grid = make_grid(transform_masks_viz(rd_gt_masks,
                                                                       self.nb_classes))
                            if self.visualizer:
                                self.visualizer.update_multi_img_masks(rd_pred_grid, rd_gt_grid,
                                                                   iteration)
            self.test_results = dict()
            self.test_results['range_doppler'] = get_metrics(rd_metrics, np.mean(rd_running_losses),
                                                             [np.mean(sub_loss) for sub_loss
                                                              in rd_running_global_losses])
            self.test_results['global_acc'] = self.test_results['range_doppler']['acc']
            self.test_results['global_prec'] = self.test_results['range_doppler']['prec']
            self.test_results['global_dice'] = self.test_results['range_doppler']['dice']
            # add global recall and mIoU zlw@20220704
            self.test_results['global_recall'] = self.test_results['range_doppler']['recall']
            self.test_results['global_miou'] = self.test_results['range_doppler']['miou']

            rd_metrics.reset()
            
        return self.test_results

    def write_params(self, path):
        """Write quantitative results of the Test"""
        with open(path, 'w') as fp:
            json.dump(self.test_results, fp)

    def set_device(self, device):
        """Set device used for test (supported: 'cuda', 'cpu')"""
        self.device = device

    def set_annot_type(self, annot_type):
        """Set annotation type to test on (specific to CARRADA)"""
        self.annot_type = annot_type
