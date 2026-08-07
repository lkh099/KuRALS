"""Main script to train KuRALSNetNPU (detection, don't train Kalman/Hungarian tracking here)"""
import argparse
import json
import numpy as np
import random
import torch
import torch.nn as nn

from kurals.utils.functions import count_params
from kurals.learners.initializer import Initializer
from kurals.legacy.model_detect import DetectionModel
from kurals.utils.distributed_utils import init_distributed_mode
from kurals.legacy.kuralsnet_npu import KuRALSNetNPU


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='Path to config file.', default='kurals/legacy/kuralsnet_npu.json')
    parser.add_argument('--dataset', default='KuRALS_CW', help='dataset for model training')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    parser.add_argument("--sync-bn", dest="sync_bn", help="Use sync batch norm", action='store_true')
    parser.add_argument("--finetune", default=False, help="the finetune path of model")
    parser.add_argument("--resume", default=None,
                         help="Path to a *_checkpoint.pt written by DetectionModel._save_checkpoint "
                              "(net + optimizer + scheduler + epoch/iteration, not just weights). "
                              "Continues training exactly (LR schedule, quantization warmup/freeze "
                              "state, best_val_loss). Note: each invocation still gets its own fresh "
                              "results/log folder from Initializer, so re-running with different "
                              "hyperparameters from the same starting checkpoint keeps each attempt's "
                              "config and logs separate -- use --finetune instead if you only want "
                              "the trained weights with a clean optimizer/schedule.")
    args = parser.parse_args()

    init_distributed_mode(args)
    with open(args.cfg, 'r') as fp:
        cfg = json.load(fp)

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
        raise KeyError('Dataset {} has not been supported yet.'.format(cfg['dataset']))

    init = Initializer(cfg)
    data = init.get_data()

    if cfg['model'] != 'kuralsnet_npu':
        raise KeyError('train_detect.py only supports model "kuralsnet_npu", got {}'.format(cfg['model']))

    net = KuRALSNetNPU(n_classes=data['cfg']['nb_classes'],
                        n_frames=data['cfg']['nb_input_channels'],
                        dataset_type=data['cfg']['dataset'])

    print('Number of trainable parameters in the model: %s' % str(count_params(net)))

    if args.distributed and args.sync_bn:
        net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)

    net.to(device)
    net.apply(_init_weights)

    if args.finetune:
        saved_model = torch.load(args.finetune, map_location=torch.device('cpu'))
        net.load_state_dict(saved_model)

    if args.distributed:
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.gpu])

    DetectionModel(net, data).train(resume_path=args.resume)


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


if __name__ == '__main__':
    main()
