"""Shared checkpoint-loading logic for KuRALSNetNPUSeg eval/export scripts
(test_kuralsnet_vs_cfar.py, export_int8.py) -- kept in one place so both agree
on how to infer quantization state from a checkpoint."""
import torch

from kurals.models import KuRALSNetNPUSeg


def load_segmentation_model(cfg, model_path, device, quant_mode='auto'):
    """Load a KuRALSNetNPUSeg from a checkpoint (a dict with a 'net' key, or a
    bare state_dict as written by kurals/learners/model.py's _save_results).

    PARAMETERS
    ----------
    cfg: dict
        Training config (needs nb_classes, nb_input_channels, dataset; optional
        bottleneck_ch, default 64).
    model_path: str or Path
    device: torch.device
    quant_mode: 'auto', 'on', or 'off'
        'auto' infers whether QAT was enabled at save time from the
        checkpoint's own 'iteration'/'cfg' fields. Falls back to True if the
        checkpoint doesn't carry that metadata (a bare state_dict), since
        every kuralsnet_npu_seg checkpoint on record was trained with QAT
        active for its full recorded schedule -- see docs/USAGE.md.

    RETURNS
    -------
    (net, quant_enabled): the loaded model (eval mode, on device) and whether
    the quantized (int8-simulated) forward path is active.
    """
    net = KuRALSNetNPUSeg(n_classes=cfg['nb_classes'], n_frames=cfg['nb_input_channels'],
                           dataset_type=cfg['dataset'], bottleneck_ch=cfg.get('bottleneck_ch', 64),
                           shallow_encoder=cfg.get('shallow_encoder', False),
                           bottleneck_kernel_size=cfg.get('bottleneck_kernel_size', 5),
                           dropout_rate=cfg.get('dropout_rate', 0))
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint['net'] if isinstance(checkpoint, dict) and 'net' in checkpoint else checkpoint
    net.load_state_dict(state_dict)
    net.to(device)
    net.eval()

    if quant_mode == 'off':
        quant_enabled = False
    elif quant_mode == 'on':
        quant_enabled = True
    else:
        quant_enabled = True
        if isinstance(checkpoint, dict) and 'iteration' in checkpoint and 'cfg' in checkpoint:
            warmup = checkpoint['cfg'].get('quant_warmup_iters')
            quant_enabled = warmup is not None and checkpoint['iteration'] >= warmup

    net.set_quant_enabled(quant_enabled)
    if quant_enabled:
        net.freeze_observers()  # no calibration drift during eval/export
    return net, quant_enabled
