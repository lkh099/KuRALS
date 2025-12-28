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
from kurals.utils import profiler
from kurals.utils.profiler import timings
from kurals.learners.tester import Tester
from kurals.models import KuRALSNet, KuRALSNet_WoASPP, KuRALSNet_ADA, KuRALSNet_PKC, KuRALSNet_AdaPKCTheta, KuRALSNet_AdaPKCXi
from kurals.models import FCN8s, UNet, deeplabv3plus_resnet101, HRNet, RSSNet
from kurals.models import SegFormer, Swin
from kurals.utils.distributed_utils import init_distributed_mode
from thop import profile
import copy

def test_model():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='kuralsnet', help='name of the model to test.')
    parser.add_argument('--dataset', default='KuRALS_CW', help='name of the dataset.')
    parser.add_argument('--mode', default=['flops', 'speed'], help='memory, flops, speed', nargs='+', type=str)
    parser.add_argument('--nb_classes', default=4, help='number of classes.', type=int)
    parser.add_argument('--nb_input_channels', default=1, help='number of input channels.', type=int)
    args = parser.parse_args()

    print(args)
    
    device = torch.device('cuda:0')
    # device = torch.device('cpu')
    if args.dataset == 'KuRALS_CW':
        args.nb_classes = 4
    else:
        args.nb_classes = 5
        
    _set_seeds()

    ## Model
    # KuRALS-Net
    if args.model == 'kuralsnet':
        model = KuRALSNet(n_classes=args.nb_classes, 
                    n_frames=args.nb_input_channels,
                   dataset_type=args.dataset)
    # KuRALS-Netw w/o ASPP
    elif args.model == 'kuralsnet_woaspp':
        model = KuRALSNet_WoASPP(n_classes=args.nb_classes, 
                    n_frames=args.nb_input_channels,
                   dataset_type=args.dataset)
    elif args.model == 'kuralsnet_ada':
        model = KuRALSNet_ADA(n_classes=args.nb_classes, 
                    n_frames=args.nb_input_channels,
                   dataset_type=args.dataset)
    elif args.model == 'kuralsnet_pkc':
        model = KuRALSNet_PKC(n_classes=args.nb_classes, 
                    n_frames=args.nb_input_channels,
                   dataset_type=args.dataset)
    elif args.model == 'kuralsnet_adapkctheta':
        model = KuRALSNet_AdaPKCTheta(n_classes=args.nb_classes, 
                    n_frames=args.nb_input_channels,
                   dataset_type=args.dataset)
    elif args.model == 'kuralsnet_adapkcxi':
        model = KuRALSNet_AdaPKCXi(n_classes=args.nb_classes, 
                    n_frames=args.nb_input_channels,
                   dataset_type=args.dataset)
    elif args.model == 'fcn8s':
        # loading parallel saved model in local single gpu mode
        model = FCN8s(n_classes=args.nb_classes, 
                    n_frames=args.nb_input_channels)
    elif args.model == 'unet':
        model = UNet(n_classes=args.nb_classes, 
                    n_frames=args.nb_input_channels)
    elif args.model == 'deeplabv3plus':
        model = deeplabv3plus_resnet101(n_classes=args.nb_classes, 
                    n_frames=args.nb_input_channels)
    elif args.model == 'hrnet':
        model = HRNet(n_classes=args.nb_classes, 
                    n_frames=args.nb_input_channels)
    elif args.model == 'rssnet':
        model = RSSNet(n_classes=args.nb_classes, 
                    n_frames=args.nb_input_channels)
    elif args.model == 'segformer':
        model = SegFormer(n_classes=args.nb_classes, 
                    n_frames=args.nb_input_channels,
                   phi='b1',
                   pretrained=False)
    elif args.model == 'swin':
        model = Swin(n_classes=args.nb_classes, 
                    n_frames=args.nb_input_channels,
                   phi='tiny',
                   pretrained=None)
    else:
        raise ValueError('model {} is not supported in test.py yet.'.format(args.model))
    
    print('Number of trainable parameters in the model: %s' % str(count_params(model)))
    
    add_temp = False
    model.eval()
    
    print(args.mode)
    
    ## Calculate Memory Consumption
    if 'memory' in args.mode:
        model.to(device)
        
        # 清空缓存，防止之前的显存占用干扰
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # 输入样例
        if args.dataset == 'KuRALS_CW':
            if args.model in ['kuralsnet', 'kuralsnet_woaspp', 'kuralsnet_ada', 'kuralsnet_pkc', 'kuralsnet_adapkctheta', 'kuralsnet_adapkcxi']:
                rd_input = torch.randn(1, 1, 1, 124, 2048).to(device)
                shape_input = (1, 1, 1, 124, 2048)
            else:
                rd_input = torch.randn(1, 1, 124, 2048).to(device)
                shape_input = (1, 1, 124, 2048)
        else:
            if args.model in ['kuralsnet', 'kuralsnet_woaspp', 'kuralsnet_ada', 'kuralsnet_pkc', 'kuralsnet_adapkctheta', 'kuralsnet_adapkcxi']:
                rd_input = torch.randn(1, 1, 1, 64, 800).to(device)
                shape_input = (1, 1, 1, 64, 800)
            else:
                rd_input = torch.randn(1, 1, 64, 800).to(device)
                shape_input = (1, 1, 64, 800)
        
        # 推理
        with torch.no_grad():
            _ = model(rd_input)

        # 当前占用显存（字节）
        current_memory = torch.cuda.memory_allocated()
        # 峰值显存占用（字节）
        peak_memory = torch.cuda.max_memory_allocated()

        print(f"Current Memory: {current_memory / 1024**2:.2f} MB")
        print(f"Peak Memory: {peak_memory / 1024**2:.2f} MB")
    
    ## Calculate FLOPs
    if 'flops' in args.mode:
        if args.dataset == 'KuRALS_CW':
            if args.model in ['kuralsnet', 'kuralsnet_woaspp', 'kuralsnet_ada', 'kuralsnet_pkc', 'kuralsnet_adapkctheta', 'kuralsnet_adapkcxi']:
                rd_input = torch.randn(1, 1, 1, 124, 2048).to(device)
                shape_input = (1, 1, 1, 124, 2048)
            else:
                rd_input = torch.randn(1, 1, 124, 2048).to(device)
                shape_input = (1, 1, 124, 2048)
        else:
            if args.model in ['kuralsnet', 'kuralsnet_woaspp', 'kuralsnet_ada', 'kuralsnet_pkc', 'kuralsnet_adapkctheta', 'kuralsnet_adapkcxi']:
                rd_input = torch.randn(1, 1, 1, 64, 800).to(device)
                shape_input = (1, 1, 1, 64, 800)
            else:
                rd_input = torch.randn(1, 1, 64, 800).to(device)
                shape_input = (1, 1, 64, 800)
        
        # Macs
        with torch.cuda.device(0):
            print('*'*20+'Measuring by thop:'+'*'*20)
            macs, params = profile(copy.deepcopy(model).to(device), (rd_input, ))
            print('macs: %.2f G, params: %.2f M' % (macs / 1e9, params / 1000000.0))
        
        # alternative way
        # with torch.cuda.device(0):
        #     print('*'*20+'Measuring by profile:'+'*'*20)
        #     macs, params = get_model_complexity_info(model, shape_input, as_strings=True, backend='pytorch', print_per_layer_stat=False, verbose=True)
        #     print('macs: ', macs, 'params: ', params)
        
        # GMACs
        # model = flopscounter.add_flops_counting_methods(model)
        # model.start_flops_count(only_conv_and_linear=True)
        # _ = model(dummy_input)
        # model.stop_flops_count()
        # print(model.total_flops_cost_repr(submodule_depth=3))
        # print('GMACS: {}'.format(model.compute_average_flops_cost()[0]/1e9))
    
    ## Prepare model
    model.to(device)
    
    ## Calculation runtime
    if 'speed' in args.mode:
        # Warm up
        print('Warmup...')
        num_warm = 200
        # model.eval()
        if args.dataset == 'KuRALS_CW':
            if args.model in ['kuralsnet', 'kuralsnet_woaspp', 'kuralsnet_ada', 'kuralsnet_pkc', 'kuralsnet_adapkctheta', 'kuralsnet_adapkcxi']:
                rd_input = torch.randn(1, 1, 1, 124, 2048).to(device)
                shape_input = (1, 1, 1, 124, 2048)
            else:
                rd_input = torch.randn(1, 1, 124, 2048).to(device)
                shape_input = (1, 1, 124, 2048)
        else:
            if args.model in ['kuralsnet', 'kuralsnet_woaspp', 'kuralsnet_ada', 'kuralsnet_pkc', 'kuralsnet_adapkctheta', 'kuralsnet_adapkcxi']:
                rd_input = torch.randn(1, 1, 1, 64, 800).to(device)
                shape_input = (1, 1, 1, 64, 800)
            else:
                rd_input = torch.randn(1, 1, 64, 800).to(device)
                shape_input = (1, 1, 64, 800)
        
        for _ in range(num_warm):
            _ = model(rd_input)
    
        # Runtime calculation with profiler
        print('Start calculating speed with profiler...')
        profiler.timings.enabled = True
        num_loops = 2
        num_speed = 500
        for loop in range(num_loops):
            timings.reset()
            for _ in range(num_speed):
                if args.dataset == 'KuRALS_CW':
                    if args.model in ['kuralsnet', 'kuralsnet_woaspp', 'kuralsnet_ada', 'kuralsnet_pkc', 'kuralsnet_adapkctheta', 'kuralsnet_adapkcxi']:
                        rd_input = torch.randn(1, 1, 1, 124, 2048).to(device)
                        shape_input = (1, 1, 1, 124, 2048)
                    else:
                        rd_input = torch.randn(1, 1, 124, 2048).to(device)
                        shape_input = (1, 1, 124, 2048)
                else:
                    if args.model in ['kuralsnet', 'kuralsnet_woaspp', 'kuralsnet_ada', 'kuralsnet_pkc', 'kuralsnet_adapkctheta', 'kuralsnet_adapkcxi']:
                        rd_input = torch.randn(1, 1, 1, 64, 800).to(device)
                        shape_input = (1, 1, 1, 64, 800)
                    else:
                        rd_input = torch.randn(1, 1, 64, 800).to(device)
                        shape_input = (1, 1, 64, 800)
                
                timings.add_count(len(rd_input))
                with torch.no_grad():
                    with timings.env(args.model):
                        _ = model(rd_input)
            
            print(f'loop: {loop}')
            print(str(profiler.timings))

def _set_seeds():
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    # torch.backends.cudnn.benchmark = False # when setting True, model will be faster but the performance will change slightly.
    # torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    test_model()
