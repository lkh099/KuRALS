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

from kurals.utils.paths import Paths
from kurals.utils.functions import count_params, normalize
from kurals.utils import flopscounter, profiler
from kurals.utils.profiler import timings
from kurals.learners.tester import Tester
from kurals.models import KuRALSNet, KuRALSNet_WoASPP, KuRALSNet_ADA, KuRALSNet_PKC, KuRALSNet_AdaPKCTheta, KuRALSNet_AdaPKCXi
from kurals.models import FCN8s, UNet, deeplabv3plus_resnet101, HRNet, RSSNet
from kurals.models import SegFormer, Swin
from kurals.loaders.dataset import KuRALS_CW, KuRALS_PD
from kurals.loaders.dataloaders import SequenceDataset, KuRALSDataset
from kurals.utils.distributed_utils import init_distributed_mode
from thop import profile
from ptflops import get_model_complexity_info
import copy


def test_model():
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

    if args.distributed:
        init_distributed_mode(args)
    print(args)
    
    cfg_path = args.cfg
    with open(cfg_path, 'r') as fp:
        cfg = json.load(fp)
    device = torch.device(cfg['device'])
    cfg['distributed'] = args.distributed
    cfg['dataset'] = args.dataset
    if cfg['dataset'] == 'KuRALS_CW':
        cfg['nb_classes'] = 4
    elif cfg['dataset'] == 'KuRALS_PD':
        cfg['nb_classes'] = 5

    _set_seeds(cfg=cfg)
    paths = Paths().get()

    ## Model
    if cfg['model'] == 'kuralsnet':
        model = KuRALSNet(n_classes=cfg['nb_classes'],
                     n_frames=cfg['nb_input_channels'],
                     dataset_type=cfg['dataset'])
    elif cfg['model'] == 'kuralsnet_woaspp':
        model = KuRALSNet_WoASPP(n_classes=cfg['nb_classes'],
                   n_frames=cfg['nb_input_channels'],
                   dataset_type=cfg['dataset'])
    elif cfg['model'] == 'kuralsnet_ada':
        model = KuRALSNet_ADA(n_classes=cfg['nb_classes'],
                   n_frames=cfg['nb_input_channels'],
                   dataset_type=cfg['dataset'])
    elif cfg['model'] == 'kuralsnet_pkc':
        model = KuRALSNet_PKC(n_classes=cfg['nb_classes'],
                     n_frames=cfg['nb_input_channels'],
                     dataset_type=cfg['dataset'])
    elif cfg['model'] == 'kuralsnet_adapkctheta':
        model = KuRALSNet_AdaPKCTheta(n_classes=cfg['nb_classes'],
                     n_frames=cfg['nb_input_channels'],
                     dataset_type=cfg['dataset'])
    elif cfg['model'] == 'kuralsnet_adapkcxi':
        model = KuRALSNet_AdaPKCXi(n_classes=cfg['nb_classes'],
                     n_frames=cfg['nb_input_channels'],
                     dataset_type=cfg['dataset'],
                     threshold=cfg['threshold'])
    elif cfg['model'] == 'fcn8s':
        model = FCN8s(n_classes=cfg['nb_classes'], 
                      n_frames=cfg['nb_input_channels'])
    elif cfg['model'] == 'unet':
        model = UNet(n_classes=cfg['nb_classes'],
                     n_frames=cfg['nb_input_channels'])
    elif cfg['model'] == 'deeplabv3plus':
        model = deeplabv3plus_resnet101(n_classes=cfg['nb_classes'],
                     n_frames=cfg['nb_input_channels'])
    elif cfg['model'] == 'hrnet':
        model = HRNet(n_classes=cfg['nb_classes'],
                   n_frames=cfg['nb_input_channels'])
    elif cfg['model'] == 'rssnet':
        model = RSSNet(n_classes=cfg['nb_classes'],
                   n_frames=cfg['nb_input_channels'])
    elif cfg['model'] == 'segformer':
        model = SegFormer(n_classes=cfg['nb_classes'],
                   n_frames=cfg['nb_input_channels'],
                   phi=cfg['phi'],
                   pretrained=False)
    elif cfg['model'] == 'swin':
        model = Swin(n_classes=cfg['nb_classes'],
                   n_frames=cfg['nb_input_channels'],
                   phi=cfg['phi'],
                   pretrained=None)
    else:
        raise ValueError('model {} is not supported in test.py yet.'.format(cfg['model']))
    
    print('Number of trainable parameters in the model: %s' % str(count_params(model)))
    
    if cfg['model'] in ['fcn8s', 'unet', 'deeplabv3plus', 'hrnet', 'rssnet', 'segformer', 'swin']:
        add_temp = False
    else:
        add_temp = True
    
    ## Calculate FLOPs
    if 'flops' in args.mode:
        if cfg['dataset'] == 'KuRALS_CW':
            if add_temp:
                dummy_input = torch.randn(1, 1, cfg['nb_input_channels'], 124, 2048)
                shape_input = (1, cfg['nb_input_channels'], 124, 2048)
            else:
                dummy_input = torch.randn(1, cfg['nb_input_channels'], 124, 2048)
                shape_input = (cfg['nb_input_channels'], 124, 2048)
        elif cfg['dataset'] == 'KuRALS_PD':
            if add_temp:
                dummy_input = torch.randn(1, 1, cfg['nb_input_channels'], 64, 800)
                shape_input = (1, cfg['nb_input_channels'], 64, 800)
            else:
                dummy_input = torch.randn(1, cfg['nb_input_channels'], 64, 800)
                shape_input = (cfg['nb_input_channels'], 64, 800)
        else:
            raise KeyError('Dataset {} has not been supported yet.'.format(cfg['dataset']))
        
        with torch.cuda.device(0):
            print('*'*20+'Measuring by profile:'+'*'*20)
            macs, params = get_model_complexity_info(model, shape_input, as_strings=True, backend='pytorch', print_per_layer_stat=False, verbose=True)
            print('macs: ', macs, 'params: ', params)
    
    ## Prepare model
    model_path = args.model_path
    saved_model = torch.load(model_path, map_location=torch.device('cpu'))
    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.load_state_dict(saved_model, strict=True)
    model.to(device)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    ## Dataset
    tester = Tester(cfg)
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
    tester.set_annot_type(cfg['annot_type'])
    
    ## Calculation runtime
    if 'speed' in args.mode:
        # Warm up
        print('Warmup...')
        num_warm = 500
        model.eval()
        if cfg['dataset'] == 'KuRALS_CW':
            if add_temp:
                dummy_input = torch.randn(1, 1, cfg['nb_input_channels'], 124, 2048)
            else:
                dummy_input = torch.randn(1, cfg['nb_input_channels'], 124, 2048)
        elif cfg['dataset'] == 'KuRALS_PD':
            if add_temp:
                dummy_input = torch.randn(1, 1, cfg['nb_input_channels'], 64, 800)
            else:
                dummy_input = torch.randn(1, cfg['nb_input_channels'], 64, 800)
        else:
            raise KeyError('Dataset {} has not been supported yet.'.format(cfg['dataset']))
        for _ in range(num_warm):
            dummy_input = dummy_input.to(device)
            _ = model(dummy_input)
    
        # Runtime calculation with profiler
        print('Start calculating speed with profiler...')
        profiler.timings.enabled = True
        num_loops = 2
        num_speed = 1000
        for loop in range(num_loops):
            timings.reset()
            for _ in range(num_speed):
                if cfg['dataset'] == 'KuRALS_CW':
                    if add_temp:
                        dummy_input = torch.randn(1, 1, cfg['nb_input_channels'], 124, 2048)
                    else:
                        dummy_input = torch.randn(1, cfg['nb_input_channels'], 124, 2048)
                elif cfg['dataset'] == 'KuRALS_PD':
                    if add_temp:
                        dummy_input = torch.randn(1, 1, cfg['nb_input_channels'], 64, 800)
                    else:
                        dummy_input = torch.randn(1, cfg['nb_input_channels'], 64, 800)
                else:
                    raise KeyError('Dataset {} has not been supported yet.'.format(cfg['dataset']))

                dummy_input = dummy_input.to(device)
                
                timings.add_count(len(dummy_input))
                with timings.env("model"):
                    _ = model(dummy_input)
            
            print(f'loop: {loop}')
            print(str(profiler.timings))
    
    ## Calculate precision, etc.
    if 'prec' in args.mode:
        print('Start calculating precision, etc.')
        test_metrics = tester.predict(model, seq_testloader, get_quali=args.get_quali, add_temp=add_temp)
        
        print('Test Prec: '
            'RD={}'.format(test_metrics['range_doppler']['prec']))
        print('Test FAR: '
            'RD={}'.format(test_metrics['range_doppler']['far']))
        print('Test mIoU: '
            'RD={}'.format(test_metrics['range_doppler']['miou']))
        print('Test mIoU by class: '
            'RD={}'.format(test_metrics['range_doppler']['miou_by_class'][0:2]))
        print(test_metrics['range_doppler']['miou_by_class'][2:])
        print('Test Dice: '
            'RD={}'.format(test_metrics['range_doppler']['dice']))
        print('Test Dice by class: '
            'RD={}'.format(test_metrics['range_doppler']['dice_by_class'][0:2]))
        print(test_metrics['range_doppler']['dice_by_class'][2:])
        print('Confusion matrix: RD=')
        
        if cfg['dataset'] == 'KuRALS_CW':
            num_class = 4
        elif cfg['dataset'] == 'KuRALS_PD':
            num_class = 5
        else:
            raise KeyError('Dataset {} has not been supported yet.'.format(cfg['dataset']))
        
        for i in range(num_class):
            print('{}'.format(test_metrics['range_doppler']['confusion_matrix'][i]))

def _set_seeds(cfg):
    np.random.seed(cfg['numpy_seed'])
    random.seed(cfg['numpy_seed'])
    torch.manual_seed(cfg['torch_seed'])
    torch.cuda.manual_seed(cfg['torch_seed'])
    torch.cuda.manual_seed_all(cfg['torch_seed'])
    torch.backends.cudnn.benchmark = False # when setting True, model will be faster but the performance will change slightly.
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    test_model()
