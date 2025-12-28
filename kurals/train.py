"""Main script to train a model"""
import argparse
import json
import os
import torch.nn as nn
import torch
import numpy as np
import random

from kurals.utils.functions import count_params
from kurals.learners.initializer import Initializer
from kurals.learners.model import Model
from kurals.utils.distributed_utils import init_distributed_mode

from kurals.models import KuRALSNet, KuRALSNet_WoASPP, KuRALSNet_ADA, KuRALSNet_PKC, KuRALSNet_AdaPKCTheta, KuRALSNet_AdaPKCXi
from kurals.models import FCN8s, UNet, deeplabv3plus_resnet101, HRNet, RSSNet
from kurals.models import SegFormer, Swin


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='Path to config file.', default='config.json')
    parser.add_argument('--dataset', default='KuRALS_CW', help='dataset for model training')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    parser.add_argument("--sync-bn", dest="sync_bn", help="Use sync batch norm", action='store_true')
    parser.add_argument("--finetune", default=False, help="the finetune path of model")
    args = parser.parse_args()

    init_distributed_mode(args)
    cfg_path = args.cfg
    with open(cfg_path, 'r') as fp:
        cfg = json.load(fp)
    
    # lt @20240525 fix the random seed.
    _set_seeds(cfg)
    
    device = torch.device(cfg['device'])
    cfg['distributed'] = args.distributed
    
    cfg['dataset'] = args.dataset
    if cfg['dataset'] == 'KuRALS_CW':
        cfg['nb_classes'] = 4
        cfg['val_step'] = 600
    elif cfg['dataset'] == 'KuRALS_PD':
        cfg['nb_classes'] = 5
    else:
        raise KeyError('Dataset {} has not been supported yet.'.format(cfg['nb_classes']))

    init = Initializer(cfg)
    data = init.get_data()
    
    if cfg['model'] == 'kuralsnet':
        net = KuRALSNet(n_classes=data['cfg']['nb_classes'],
                     n_frames=data['cfg']['nb_input_channels'],
                     dataset_type=data['cfg']['dataset'])
    elif cfg['model'] == 'kuralsnet_woaspp':
        net = KuRALSNet_WoASPP(n_classes=data['cfg']['nb_classes'],
                   n_frames=data['cfg']['nb_input_channels'],
                   dataset_type=data['cfg']['dataset'])
    elif cfg['model'] == 'kuralsnet_ada':
        net = KuRALSNet_ADA(n_classes=data['cfg']['nb_classes'],
                   n_frames=data['cfg']['nb_input_channels'],
                   dataset_type=data['cfg']['dataset'])
    elif cfg['model'] == 'kuralsnet_pkc':
        net = KuRALSNet_PKC(n_classes=data['cfg']['nb_classes'],
                     n_frames=data['cfg']['nb_input_channels'],
                     dataset_type=data['cfg']['dataset'])
    elif cfg['model'] == 'kuralsnet_adapkctheta':
        net = KuRALSNet_AdaPKCTheta(n_classes=data['cfg']['nb_classes'],
                     n_frames=data['cfg']['nb_input_channels'],
                     dataset_type=data['cfg']['dataset'])
    elif cfg['model'] == 'kuralsnet_adapkcxi':
        net = KuRALSNet_AdaPKCXi(n_classes=data['cfg']['nb_classes'],
                     n_frames=data['cfg']['nb_input_channels'],
                     dataset_type=data['cfg']['dataset'],
                     threshold=data['cfg']['threshold'])
    elif cfg['model'] == 'fcn8s':
        net = FCN8s(n_classes=data['cfg']['nb_classes'],
                    n_frames=data['cfg']['nb_input_channels'])
    elif cfg['model'] == 'unet':
        net = UNet(n_classes=data['cfg']['nb_classes'],
                   n_frames=data['cfg']['nb_input_channels'])
    elif cfg['model'] == 'deeplabv3plus':
        net = deeplabv3plus_resnet101(n_classes=data['cfg']['nb_classes'],
                                      n_frames=data['cfg']['nb_input_channels'])
    elif cfg['model'] == 'hrnet':
        net = HRNet(n_classes=data['cfg']['nb_classes'],
                   n_frames=data['cfg']['nb_input_channels'])
    elif cfg['model'] == 'rssnet':
        net = RSSNet(n_classes=data['cfg']['nb_classes'],
                   n_frames=data['cfg']['nb_input_channels'])
    elif cfg['model'] == 'segformer':
        net = SegFormer(n_classes=data['cfg']['nb_classes'],
                   n_frames=data['cfg']['nb_input_channels'],
                   phi=data['cfg']['phi'],
                   pretrained=False)
    elif cfg['model'] == 'swin':
        net = Swin(n_classes=data['cfg']['nb_classes'],
                   n_frames=data['cfg']['nb_input_channels'],
                   phi=data['cfg']['phi'],
                   pretrained=None)
    else:
        raise KeyError('Model {} has not been supported yet.'.format(cfg['model']))

    print('Number of trainable parameters in the model: %s' % str(count_params(net)))

    if args.distributed and args.sync_bn:
        net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
    
    net.to(device)
    net.apply(_init_weights)
    
    if args.finetune:
        saved_model = torch.load(args.finetune, map_location=torch.device('cpu'))
        net.load_state_dict(saved_model)

    net_without_ddp = net
    if args.distributed:
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.gpu])
        net_without_ddp = net.module
    
    if cfg['model'] in ['fcn8s', 'unet', 'deeplabv3plus', 'hrnet', 'rssnet', 'segformer', 'swin']:
        Model(net, data).train(add_temp=False)
    else:
        Model(net, data).train(add_temp=True)

def _init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.)
    elif isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.uniform_(m.weight, 0., 1.)
            nn.init.constant_(m.bias, 0.)

def _set_seeds(cfg):
    np.random.seed(cfg['numpy_seed'])
    random.seed(cfg['numpy_seed'])
    torch.manual_seed(cfg['torch_seed'])
    torch.cuda.manual_seed(cfg['torch_seed'])
    torch.cuda.manual_seed_all(cfg['torch_seed'])
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    main()
