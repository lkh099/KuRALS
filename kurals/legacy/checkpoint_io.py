"""Shared checkpoint-loading logic for the retired KuRALSNetNPU (CenterNet-head)
eval scripts in this package (test_detect_vs_cfar.py, test_class_discrimination.py).
Superseded for the live model by kurals/utils/checkpoint_io.py."""
import torch

from kurals.legacy.kuralsnet_npu import KuRALSNetNPU


def load_detection_model(cfg, model_path, device, quant_mode='auto'):
    """Load a KuRALSNetNPU from a checkpoint (a dict with a 'net' key, as
    written by DetectionModel._save_checkpoint) or a bare state_dict.

    PARAMETERS
    ----------
    cfg: dict
        Training config (needs nb_classes, nb_input_channels, dataset).
    model_path: str or Path
    device: torch.device
    quant_mode: 'auto', 'on', or 'off'
        'auto' infers whether QAT was enabled at save time from the
        checkpoint's own 'iteration'/'cfg' fields, the same way
        DetectionModel.train()'s resume path does. Falls back to False if the
        checkpoint doesn't carry that metadata (e.g. a bare state_dict).

    RETURNS
    -------
    (net, quant_enabled): the loaded model (eval mode, on device) and whether
    the quantized (int8-simulated) forward path is active.
    """
    net = KuRALSNetNPU(n_classes=cfg['nb_classes'], n_frames=cfg['nb_input_channels'],
                        dataset_type=cfg['dataset'])
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
        quant_enabled = False
        if isinstance(checkpoint, dict) and 'iteration' in checkpoint and 'cfg' in checkpoint:
            warmup = checkpoint['cfg'].get('quant_warmup_iters')
            if warmup is not None and checkpoint['iteration'] >= warmup:
                quant_enabled = True

    net.set_quant_enabled(quant_enabled)
    if quant_enabled:
        net.freeze_observers()  # no calibration drift during eval/export
    return net, quant_enabled
