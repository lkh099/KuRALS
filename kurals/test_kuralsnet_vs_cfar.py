"""Compare a trained (non-NPU-constrained) segmentation model -- KuRALSNet by
default -- against CFAR using the exact same binary foreground/background
convention as test_cfar.py and test_detect_vs_cfar.py: ground truth is
(1 - rd_mask[:, 0]) ("not background"), prediction is a binary foreground mask,
scored with Evaluator(num_class=2) so Precision/Pd/FAR/mIoU/Dice are directly
comparable across all three (CFAR, the NPU detector, and this model).

KuRALSNet outputs per-pixel logits over all n_classes (background + fg classes)
at the native input resolution (no separate decode step, unlike the anchor-free
NPU detector), so the foreground prediction is simply argmax(logits, dim=1) != 0.
"""
import argparse
import json

import torch
from torch.utils.data import DataLoader

from kurals.models import (KuRALSNet, KuRALSNet_WoASPP, KuRALSNet_ADA, KuRALSNet_PKC,
                            KuRALSNet_AdaPKCTheta, FCN8s, UNet, deeplabv3plus_resnet101,
                            HRNet, RSSNet, SegFormer, Swin, KuRALSNetNPUSeg)
from kurals.models.cfar import CFAR2D_Parallel
from kurals.loaders.dataset import KuRALS_CW, KuRALS_PD
from kurals.loaders.dataloaders import SequenceDataset, KuRALSDataset
from kurals.utils.functions import normalize
from kurals.utils.metrics import Evaluator
from kurals.utils.paths import Paths

MODEL_CTORS = {
    'kuralsnet': lambda nc, nf, dt, bc=64: KuRALSNet(n_classes=nc, n_frames=nf, dataset_type=dt),
    'kuralsnet_woaspp': lambda nc, nf, dt, bc=64: KuRALSNet_WoASPP(n_classes=nc, n_frames=nf, dataset_type=dt),
    'kuralsnet_ada': lambda nc, nf, dt, bc=64: KuRALSNet_ADA(n_classes=nc, n_frames=nf, dataset_type=dt),
    'kuralsnet_pkc': lambda nc, nf, dt, bc=64: KuRALSNet_PKC(n_classes=nc, n_frames=nf, dataset_type=dt),
    'kuralsnet_adapkctheta': lambda nc, nf, dt, bc=64: KuRALSNet_AdaPKCTheta(n_classes=nc, n_frames=nf, dataset_type=dt),
    'fcn8s': lambda nc, nf, dt, bc=64: FCN8s(n_classes=nc, n_frames=nf),
    'unet': lambda nc, nf, dt, bc=64: UNet(n_classes=nc, n_frames=nf),
    'deeplabv3plus': lambda nc, nf, dt, bc=64: deeplabv3plus_resnet101(n_classes=nc, n_frames=nf),
    'hrnet': lambda nc, nf, dt, bc=64: HRNet(n_classes=nc, n_frames=nf),
    'rssnet': lambda nc, nf, dt, bc=64: RSSNet(n_classes=nc, n_frames=nf),
    'kuralsnet_npu_seg': lambda nc, nf, dt, bc=64, **kw: KuRALSNetNPUSeg(n_classes=nc, n_frames=nf, dataset_type=dt, bottleneck_ch=bc, **kw),
}
ADD_TEMP_MODELS = {'kuralsnet', 'kuralsnet_woaspp', 'kuralsnet_ada', 'kuralsnet_pkc', 'kuralsnet_adapkctheta'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', required=True)
    parser.add_argument('--model-path', dest='model_path', required=True)
    parser.add_argument('--dataset', default='KuRALS_CW')
    parser.add_argument('--split', default='Test')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--cfar-alphas', dest='cfar_alphas', type=float, nargs='+', default=[0.5, 1.0, 1.5])
    args = parser.parse_args()

    with open(args.cfg, 'r') as fp:
        cfg = json.load(fp)
    cfg['dataset'] = args.dataset
    cfg['nb_classes'] = 4 if args.dataset == 'KuRALS_CW' else 5

    device = torch.device(args.device)
    paths = Paths().get()

    add_temp = cfg['model'] in ADD_TEMP_MODELS
    # kuralsnet_npu_seg's remaining ctor args reshape the architecture (kernel size) or
    # its resolution ladder (encoder_depth/stem_stride) -- a checkpoint trained with any
    # of them set won't load without passing them here too.
    seg_kwargs = dict(shallow_encoder=cfg.get('shallow_encoder', False),
                      bottleneck_kernel_size=cfg.get('bottleneck_kernel_size', 5),
                      dropout_rate=cfg.get('dropout_rate', 0),
                      encoder_depth=cfg.get('encoder_depth', None),
                      stem_stride=cfg.get('stem_stride', None)) if cfg['model'] == 'kuralsnet_npu_seg' else {}
    net = MODEL_CTORS[cfg['model']](cfg['nb_classes'], cfg['nb_input_channels'], cfg['dataset'],
                                    cfg.get('bottleneck_ch', 64), **seg_kwargs)
    state = torch.load(args.model_path, map_location='cpu')
    net.load_state_dict(state, strict=True)
    net.to(device)
    net.eval()
    # kurals/learners/model.py's _save_results writes a bare state_dict with no
    # quant-state metadata (unlike DetectionModel's checkpoints), and this script
    # never enabled QAT itself -- for a QAT-trained NPU-legal model (currently
    # only kuralsnet_npu_seg), evaluating without this would silently score the
    # float32-equivalent network, not the actual int8-simulated NPU behavior.
    # Every other registered model here is float-only and lacks this method.
    if hasattr(net, 'set_quant_enabled'):
        net.set_quant_enabled(True)
        net.freeze_observers()  # no calibration drift during eval

    data = (KuRALS_CW() if args.dataset == 'KuRALS_CW' else KuRALS_PD()).get(args.split)
    seq_loader = DataLoader(SequenceDataset(data), batch_size=1, shuffle=False, num_workers=0)

    model_evaluator = Evaluator(num_class=2)
    cfar_models = {alpha: CFAR2D_Parallel(cfar_type='CA', alpha=alpha, ref_cells=1, guard_cells=1)
                   for alpha in args.cfar_alphas}
    cfar_evaluators = {alpha: Evaluator(num_class=2) for alpha in args.cfar_alphas}

    with torch.no_grad():
        for seq_name, seq in seq_loader:
            path_to_frames = paths[args.dataset] / seq_name[0]
            kurals_dataset = KuRALSDataset(seq, cfg['annot_type'], path_to_frames,
                                            cfg['process_signal'], cfg['nb_input_channels'],
                                            None, add_temp)
            frame_dataloader = DataLoader(kurals_dataset, shuffle=False, batch_size=cfg['batch_size'], num_workers=0)

            for frame in frame_dataloader:
                rd_data = frame['rd_matrix'].to(device).float()
                rd_mask = frame['rd_mask'].float()
                rd_data_norm = normalize(rd_data, cfg['dataset'], 'range_doppler', norm_type=cfg['norm_type'])
                gt_fg = (1 - rd_mask[:, 0, :, :]).int()

                logits = net(rd_data_norm).cpu()
                pred_fg = (torch.argmax(logits, dim=1) != 0).int()
                model_evaluator.add_batch(gt_fg, pred_fg)

                rd_data_cpu = rd_data.cpu()
                for alpha, cfar in cfar_models.items():
                    cfar_out = cfar.filter(rd_data_cpu.squeeze(1) if add_temp else rd_data_cpu)
                    cfar_evaluators[alpha].add_batch(gt_fg, cfar_out.squeeze(1))

    def report(name, evaluator):
        acc, _ = evaluator.get_pixel_acc_class()
        prec, _ = evaluator.get_pixel_prec_class()
        _, recall_by_class = evaluator.get_pixel_recall_class()
        _, far_by_class = evaluator.get_pixel_far_class()
        miou, _ = evaluator.get_miou_class()
        dice, _ = evaluator.get_dice_class()
        print(f'{name:>18}  Acc={acc:.4f}  Prec={prec:.4f}  Pd={recall_by_class[1]:.4f}  '
              f'FAR={far_by_class[1]:.4f}  mIoU={miou:.4f}  mDice={dice:.4f}')

    print(f'\n=== {cfg["model"]} vs CFAR on {args.dataset} / {args.split} (binary fg/bg) ===')
    report(cfg['model'], model_evaluator)
    for alpha, evaluator in cfar_evaluators.items():
        report(f'CFAR(CA) a={alpha}', evaluator)


if __name__ == '__main__':
    main()
