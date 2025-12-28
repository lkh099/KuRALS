"""Main script to test a pretrained model"""
import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import random
from tqdm import tqdm
import time
import os

from kurals.utils.paths import Paths
from kurals.utils.functions import count_params, normalize
from kurals.utils import flopscounter, profiler
from kurals.utils.profiler import timings
from kurals.learners.tester import Tester
from kurals.models.cfar import CFAR2D_Parallel
from kurals.loaders.dataset import KuRALS_CW, KuRALS_PD
from kurals.loaders.dataloaders import SequenceDataset, KuRALSDataset
from kurals.utils.distributed_utils import init_distributed_mode
from thop import profile
from ptflops import get_model_complexity_info
import copy
from kurals.utils.metrics import Evaluator

def test_cfar():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='config.json', help='Path to config file of the model to test.')
    parser.add_argument('--dataset', default='KuRALS_CW', help='dataset for model training')
    parser.add_argument('--mode', default=['prec'], help='flops, prec, speed', nargs='+', type=str)
    parser.add_argument('--model-path', default=None, dest="model_path", help='Path to pretrained model weight.')
    parser.add_argument('--split', default='Test', help='Train, Validation or Test dataset')
    parser.add_argument("--get-quali", dest="get_quali", action='store_true', help="Output predicted images")
    parser.add_argument('--speed-percent', dest="speed_percent", default=1.0, type=float, help='Percentage of dataset in calculating speed')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    parser.add_argument("--distributed", action='store_true', help="Use distributed testing")
    parser.add_argument("--sync-bn", dest="sync_bn", action='store_true', help="Use sync batch norm")
    args = parser.parse_args()

    print(args)
    
    cfg_path = args.cfg
    with open(cfg_path, 'r') as fp:
        cfg = json.load(fp)

    cfg['distributed'] = args.distributed
    cfg['dataset'] = args.dataset
    if cfg['dataset'] == 'KuRALS_CW':
        cfg['nb_classes'] = 4
    elif cfg['dataset'] == 'KuRALS_PD':
        cfg['nb_classes'] = 5

    _set_seeds(cfg=cfg)
    paths = Paths().get()

    ## Model
    cfar_type = 'CA'
    print(f'cfar type: {cfar_type}')
    arr_pd = []
    arr_pfa = []
    for alpha in np.arange(0.5, 2, 0.5):
        refer_cell = 1
        guard_cell = 1
        add_temp = False
        nb_input_channels = 1
        num_class = 2 # 2 for cfar method
        
        model = CFAR2D_Parallel(cfar_type=cfar_type, alpha=alpha, ref_cells=refer_cell, guard_cells=guard_cell)
        
        ## Dataset
        # print(tester)
        if cfg['dataset'] == 'KuRALS_CW':
            data = KuRALS_CW()
        elif cfg['dataset'] == 'KuRALS_PD':
            data = KuRALS_PD()
        else:
            raise KeyError('Dataset {} has not been supported yet.'.format(cfg['dataset']))
        
        test = data.get(args.split)
        testset = SequenceDataset(test)
        seq_testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)
        
        rd_metrics = Evaluator(num_class=num_class)
        
        for sequence_data in seq_testloader:
            # print('processing: {} / {}'.format(i, len(seq_loader)))
            seq_name, seq = sequence_data
            if cfg['dataset'] == 'KuRALS_CW':
                path_to_frames = paths['KuRALS_CW'] / seq_name[0]
            elif cfg['dataset'] == 'KuRALS_PD':
                path_to_frames = paths['KuRALS_PD'] / seq_name[0]
            else:
                raise KeyError(f'Dataset has not been supported yet.')
            
            kurals_dataset = KuRALSDataset(seq,
                                            cfg['annot_type'],
                                            path_to_frames,
                                            cfg['process_signal'],
                                            cfg['nb_input_channels'],
                                            None,
                                            add_temp)
            
            sampler_test = torch.utils.data.SequentialSampler(kurals_dataset)
            # lt@20240301 for pdr dataset, fix batch size to 2.
            if cfg['dataset'] == 'KuRALS_CW':
                frame_dataloader = DataLoader(kurals_dataset,
                                            shuffle=False,
                                            sampler=sampler_test,
                                            batch_size=cfg['batch_size'],
                                            num_workers=cfg['num_workers'])
            elif cfg['dataset'] == 'KuRALS_PD':
                frame_dataloader = DataLoader(kurals_dataset,
                                            shuffle=False,
                                            sampler=sampler_test,
                                            batch_size=1,
                                            num_workers=0)
            else:
                raise KeyError(f'Dataset has not been supported yet.')
            
            for j, frame in enumerate(frame_dataloader):
                rd_data = frame['rd_matrix'].float()
                rd_mask = frame['rd_mask'].float()
                rd_data = normalize(rd_data, cfg['dataset'], 'range_doppler', norm_type=cfg['norm_type'])
                
                rd_outputs = model.filter(rd_data)
                rd_metrics.add_batch((1-rd_mask[:, 0, :, :]).int(), rd_outputs.squeeze(1))
                

        acc, acc_by_class = rd_metrics.get_pixel_acc_class()  # harmonic_mean=True)
        prec, prec_by_class = rd_metrics.get_pixel_prec_class()
        recall, recall_by_class = rd_metrics.get_pixel_recall_class()  # harmonic_mean=True)
        far, far_by_class = rd_metrics.get_pixel_far_class()
        miou, miou_by_class = rd_metrics.get_miou_class()  # harmonic_mean=True)
        dice, dice_by_class = rd_metrics.get_dice_class()
        confusion_matrix = rd_metrics.get().tolist()
        
        print(f'alpha      : {alpha}')
        print(f'Acc        : {acc}')
        print(f'Precision  : {prec}')
        print(f'Recall/Pd  : {recall_by_class[1]}')
        print(f'FAR        : {far_by_class[1]}')
        print(f'mIoU       : {miou}')
        print(f'mDice       : {dice}')
        print(f'confusion_matrix     : {confusion_matrix}')
        
        arr_pd.append(round(recall_by_class[1], 3))
        arr_pfa.append(round(far_by_class[1], 3))
    
    print('----------------------------------')
    print('PD array : ', arr_pd)
    print('PFA array: ', arr_pfa)

def _set_seeds(cfg):
    np.random.seed(cfg['numpy_seed'])
    random.seed(cfg['numpy_seed'])
    torch.manual_seed(cfg['torch_seed'])
    torch.cuda.manual_seed(cfg['torch_seed'])
    torch.cuda.manual_seed_all(cfg['torch_seed'])
    torch.backends.cudnn.benchmark = False # when setting True, model will be faster but the performance will change slightly.
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    test_cfar()
