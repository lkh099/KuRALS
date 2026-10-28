"""A lot of functions used in our pipelines"""
import json
import os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from fnmatch import fnmatch

from kurals.utils import KURAD_HOME
from kurals.losses.ce_loss import CELoss
from kurals.losses.soft_dice import SoftDiceLoss
from kurals.losses.generalized_dice import GeneralizedDiceLoss
from kurals.losses.focal_loss import FocalLoss
from kurals.losses.nbs_loss import NBSLoss
from kurals.losses.nbsfocal import NBSFocalLoss
from kurals.loaders.dataloaders import Rescale, Flip, HFlip, VFlip, GainJitter, NoiseJitter


def get_class_weights(dataset_type, signal_type):
    """Load class weights for custom loss

    PARAMETERS
    ----------
    signal_type: str
        Supported: 'range_doppler', 'range_angle'

    RETURNS
    -------
    weights: numpy array
    """
    weight_path = KURAD_HOME / 'config_files'
    
    if dataset_type == 'KuRALS_CW':
        file_name = 'cwr'
    elif dataset_type == 'KuRALS_PD':
        file_name = 'pdr'
    else:
        raise ValueError('Dataset type {} is not supported.'.format(dataset_type))
    
    if signal_type in ('range_angle'):
        file_name = file_name + '_' + 'ra_weights.json'
    elif signal_type in ('range_doppler'):
        file_name = file_name + '_' + 'rd_weights.json'
    else:
        raise ValueError('Signal type {} is not supported.'.format(signal_type))
    file_path = weight_path / file_name
    with open(file_path, 'r') as fp:
        weights = json.load(fp)
    
    if dataset_type == 'KuRALS_CW':
        weights = np.array([weights['Background'], weights['UAV'],
                        weights['Pedestrian'], weights['Vehicle']])
    elif dataset_type == 'KuRALS_PD':
        weights = np.array([weights['Background'], weights['UAV'],
                        weights['Pedestrian'], weights['Car'],
                        weights['Boat']])
    else:
        raise ValueError('Dataset type {} is not supported.'.format(dataset_type))
    # weights = np.array([weights['background'], weights['UAV'],
    #                     weights['Pedestrian'], weights['Vehicle']])
    weights = torch.from_numpy(weights)
    return weights

# zlw@20220302 Balance the RD and RA Losses
def get_loss_weight(signal_type):
    """Load weight for rd and ra loss
    PARAMETERS
    ----------
    signal_type: str
        Supported: 'range_doppler', 'range_angle'

    RETURNS
    -------
    weight: numpy float
    """
    if signal_type in ('range_angle'):
        weight = 2.0
    elif signal_type in ('range_doppler'):
        weight = 1.0
    else:
        raise ValueError('Signal type {} is not supported.'.format(signal_type))
    return weight

def transform_masks_viz(masks, nb_classes):
    """Used for visualization"""
    masks = masks.unsqueeze(1)
    masks = (masks.float()/nb_classes)
    return masks


def get_metrics(metrics, loss, losses=None):
    """Structure the metric results

    PARAMETERS
    ----------
    metrics: object
        Contains statistics recorded during inference
    loss: tensor
        Loss value
    losses: list
        List of loss values

    RETURNS
    -------
    metrics_values: dict
    """
    metrics_values = dict()
    metrics_values['loss'] = loss.item()
    if isinstance(losses, list):
        metrics_values['loss_ce'] = losses[0].item()
        metrics_values['loss_dice'] = losses[1].item()
    acc, acc_by_class = metrics.get_pixel_acc_class()  # harmonic_mean=True)
    prec, prec_by_class = metrics.get_pixel_prec_class()
    recall, recall_by_class = metrics.get_pixel_recall_class()  # harmonic_mean=True)
    miou, miou_by_class = metrics.get_miou_class()  # harmonic_mean=True)
    dice, dice_by_class = metrics.get_dice_class()
    far, far_by_class = metrics.get_pixel_far_class()
    metrics_values['acc'] = acc
    metrics_values['acc_by_class'] = acc_by_class.tolist()
    metrics_values['prec'] = prec
    metrics_values['prec_by_class'] = prec_by_class.tolist()
    metrics_values['recall'] = recall
    metrics_values['recall_by_class'] = recall_by_class.tolist()
    metrics_values['miou'] = miou
    metrics_values['miou_by_class'] = miou_by_class.tolist()
    metrics_values['dice'] = dice
    metrics_values['dice_by_class'] = dice_by_class.tolist()
    # lt@20251029 add false alarm rate
    metrics_values['far'] = far
    metrics_values['far_by_class'] = far_by_class.tolist()
    # lt@20240311 add confusion matrix
    conf_matrix = metrics.get()
    metrics_values['confusion_matrix'] = conf_matrix.tolist()
    return metrics_values


# ---------------------------------------------------------------------------
# DSP -> NPU input contract ("dsp_log2_5p3"), from npu_riscv's
# docs/integration/radar_dsp_npu_input.md. The radar DSP hardware turns each uint32
# range-Doppler power value P into one int8 pixel:
#     log8(P) = {leading_one_index(P), three_bits_below_leading_one(P)}   (5.3-bit log2)
#     q       = clamp(log8(P) - LOG_SCALE, 0, 127)
# and the NPU consumes q directly (input_scale == 1.0, natural FFT Doppler order).
# KuRALS stores float64 RD magnitudes, so P is emulated exactly as npu_riscv's
# "proxy bridge" does: P = uint32(floor(value)) >> proxy_power_right_shift (default 16).
# norm_type syntax: 'dsp_log' | 'dsp_log:<proxy_shift>' | 'dsp_log:<proxy_shift>:<log_scale>'.
# ---------------------------------------------------------------------------
DSP_LOG_DEFAULT_SHIFT = 16
DSP_LOG_DEFAULT_SCALE = 0


def is_dsp_log(norm_type):
    return isinstance(norm_type, str) and norm_type.split(':')[0] == 'dsp_log'


def parse_dsp_log(norm_type):
    parts = norm_type.split(':')
    shift = int(parts[1]) if len(parts) > 1 else DSP_LOG_DEFAULT_SHIFT
    scale = int(parts[2]) if len(parts) > 2 else DSP_LOG_DEFAULT_SCALE
    assert 0 <= scale <= 255, 'LOG_SCALE is a uint8 APB field'
    return shift, scale


def dsp_log8(power_u32):
    """Bit-exact 5.3 log2 code of integer power (int64 tensor holding uint32 values), as
    power_quant_log2.sv / sim/e2e_golden/dsp_multiscale.log8_uint32. 0 -> 0."""
    p = power_u32.long()
    exponent = torch.zeros_like(p)
    for bit in range(32):
        exponent = torch.where((p >> bit) & 1 == 1, torch.full_like(p, bit), exponent)
    up = (p << (3 - exponent).clamp(min=0)) & 7          # exponent < 3: pad zeros below
    down = (p >> (exponent - 3).clamp(min=0)) & 7        # exponent >= 3: take the 3 bits under the MSB
    mantissa = torch.where(exponent >= 3, down, up)
    code = (exponent << 3) | mantissa
    return torch.where(p == 0, torch.zeros_like(code), code)


def dsp_log_quantize(data, shift=DSP_LOG_DEFAULT_SHIFT, scale=DSP_LOG_DEFAULT_SCALE):
    """float RD magnitudes -> the int8 pixel codes the DSP hands the NPU, returned as float
    tensors holding integers in [0, 127] (so the stem sees them with input_scale 1.0)."""
    p = torch.floor(data.double()).clamp(0, 2 ** 32 - 1).long() >> shift
    q = (dsp_log8(p) - scale).clamp(0, 127)
    return q.to(data.dtype if data.is_floating_point() else torch.float32)


def to_natural_doppler(x, norm_type):
    """The DSP/NPU contract uses natural FFT Doppler order (zero-Doppler at index 0); the
    dataset stores it fftshifted (zero-Doppler centred). Apply ifftshift on the Doppler axis
    (dim -2) to both RD input and mask so the network trains on the deployed layout. No-op
    for other norm types, which keep the legacy layout."""
    if is_dsp_log(norm_type) and not os.environ.get('KURALS_NO_ROLL'):  # env switch: diagnostics only
        return torch.roll(x, shifts=-(x.shape[-2] // 2), dims=-2)
    return x


def normalize(data, dataset_type, signal_type, norm_type='local'):
    """
    Method to normalise the radar views

    PARAMETERS
    ----------
    data: numpy array
        Radar view (batch)
    signal_type: str
        Type of radar view
        Supported: 'range_doppler', 'range_angle' and 'angle_doppler'
    norm_type: str
        Type of normalisation to apply
        Supported: 'local', 'tvt'

    RETURNS
    -------
    norm_data: numpy array
        normalised radar view
    """
    if is_dsp_log(norm_type):
        if signal_type != 'range_doppler':
            raise TypeError('dsp_log is only defined for range_doppler')
        return dsp_log_quantize(data, *parse_dsp_log(norm_type))

    if dataset_type == 'KuRALS_CW':
        file_name = 'cwr'
    elif dataset_type == 'KuRALS_PD':
        file_name = 'pdr'
    else:
        raise ValueError('Dataset type {} is not supported.'.format(dataset_type))
    
    if norm_type in ('local'):
        min_value = torch.min(data)
        max_value = torch.max(data)
        norm_data = torch.div(torch.sub(data, min_value), torch.sub(max_value, min_value))
        return norm_data

    elif signal_type == 'range_doppler':
        if norm_type in ('tvt', 'std'):
            file_path = KURAD_HOME / 'config_files' / (file_name + '_' + 'rd_stats_all.json')
        else:
            raise TypeError('Global type {} is not supported'.format(norm_type))
        with open(file_path, 'r') as fp:
            rd_stats = json.load(fp)
        min_value = torch.tensor(rd_stats['min_val'])
        max_value = torch.tensor(rd_stats['max_val'])
        mean_value = torch.tensor(rd_stats['mean'])
        std_value = torch.tensor(rd_stats['std'])

    elif signal_type == 'range_angle':
        if norm_type in ('tvt', 'std'):
            file_path = KURAD_HOME / 'config_files' / (file_name + '_' + 'ra_stats_all.json')
        else:
            raise TypeError('Global type {} is not supported'.format(norm_type))
        with open(file_path, 'r') as fp:
            ra_stats = json.load(fp)
        min_value = torch.tensor(ra_stats['min_val'])
        max_value = torch.tensor(ra_stats['max_val'])
        mean_value = torch.tensor(rd_stats['mean'])
        std_value = torch.tensor(rd_stats['std'])

    elif signal_type == 'angle_doppler':
        if norm_type in ('tvt', 'std'):
            file_path = KURAD_HOME / 'config_files' / (file_name + '_' + 'ad_stats_all.json')
        else:
            raise TypeError('Global type {} is not supported'.format(norm_type))
        with open(file_path, 'r') as fp:
            ad_stats = json.load(fp)
        min_value = torch.tensor(ad_stats['min_val'])
        max_value = torch.tensor(ad_stats['max_val'])
        mean_value = torch.tensor(rd_stats['mean'])
        std_value = torch.tensor(rd_stats['std'])

    else:
        raise TypeError('Signal {} is not supported.'.format(signal_type))

    if norm_type == 'tvt':
        norm_data = torch.div(torch.sub(data, min_value),
                            torch.sub(max_value, min_value))
    elif norm_type == 'std':
        norm_data = torch.div(torch.sub(data, mean_value), std_value)
    return norm_data


def define_loss(dataset_type, signal_type, custom_loss, device, label_smoothing=0):
    """
    Method to define the loss to use during training

    PARAMETERS
    ----------
    signal_type: str
        Type of radar view
        Supported: 'range_doppler', 'range_angle' or 'angle_doppler'
    custom loss: str
        Short name of the custom loss to use
        Supported: 'wce', 'sdice', 'wce_w10sdice' or 'wce_w10sdice_w5col'
        Default: Cross Entropy is used for any other str
    devide: str
        Supported: 'cuda' or 'cpu'
    """
    if dataset_type == 'KuRALS_CW':
        num_classes = 4
        threshold_nbs = 0.6
    elif dataset_type == 'KuRALS_PD':
        num_classes = 5
        threshold_nbs = 0.2
    else:
        raise KeyError('Dataset {} has not been supported yet.'.format(num_classes))
    
    if custom_loss == 'ce':
        celoss = CELoss(num_classes=num_classes)
        loss = [celoss, lambda x, y: torch.tensor(0., device=x.device)]
    elif custom_loss == 'wce':
        weights = get_class_weights(dataset_type, signal_type)
        loss = [CELoss(weight=weights.to(device).float()), lambda x, y: torch.tensor(0., device=x.device)]
    elif custom_loss == 'focal':
        focaloss = FocalLoss(weight=None, num_classes=num_classes)
        loss = [focaloss, lambda x, y: torch.tensor(0., device=x.device)]
    elif custom_loss == 'nbs1':
        nbsloss = NBSLoss(weight=None, threshold=threshold_nbs, degree=1, num_classes=num_classes, gamma=1,
                           label_smoothing=label_smoothing)
        loss = [nbsloss, lambda x, y: torch.tensor(0., device=x.device)]
    elif custom_loss == 'nbs1_wce':
        # Same NBS1 recalibration as 'nbs1' (noisy/hard-background suppression),
        # but the underlying per-pixel CE is class-weighted using this dataset's
        # own precomputed inverse-frequency weights (get_class_weights ->
        # config_files/cwr_rd_weights.json for KuRALS_CW: Background~1e-6,
        # UAV~0.051, Pedestrian~0.083, Vehicle~0.866) instead of NBSLoss's
        # default weight=None. Every other custom_loss option that wants class
        # weighting already uses this exact weight source (see 'wce',
        # 'wce_w10sdice', 'wfocal_w*sdice' above) -- 'nbs1'/'nbs2' were the only
        # NBS variants that never combined it with NBS's own background
        # suppression mechanism, even though the two address different problems
        # (NBS discounts *hard* background pixels regardless of class; the
        # weights address the *class imbalance* baked into the CE loss itself)
        # and are not redundant.
        weights = get_class_weights(dataset_type, signal_type)
        nbsloss = NBSLoss(weight=weights.to(device).float(), threshold=threshold_nbs, degree=1,
                           num_classes=num_classes, gamma=1)
        loss = [nbsloss, lambda x, y: torch.tensor(0., device=x.device)]
    elif custom_loss == 'nbs2':
        nbsloss = NBSLoss(weight=None, threshold=threshold_nbs, degree=2, num_classes=num_classes, gamma=1)
        loss = [nbsloss, lambda x, y: torch.tensor(0., device=x.device)]
    elif custom_loss == 'nbsfocal':
        sfocalloss = NBSFocalLoss(weight=None, threshold=threshold_nbs, degree=2, num_classes=num_classes, 
                               gamma=2, down_index=0)
        loss = [sfocalloss, lambda x, y: torch.tensor(0., device=x.device)]
    elif custom_loss == 'sdice':
        loss = [SoftDiceLoss(), lambda x, y: torch.tensor(0., device=x.device)]
    elif custom_loss == 'gendice':
        loss = [GeneralizedDiceLoss(squared_pred=True, smooth=1e-8, batch=False, w_type='square'), lambda x, y: torch.tensor(0., device=x.device)]
    elif custom_loss == 'ce_sdice':
        loss = [nn.CrossEntropyLoss(), SoftDiceLoss(global_weight=10.)]
    elif custom_loss == 'wce_w10sdice':
        weights = get_class_weights(dataset_type, signal_type)
        ce_loss = nn.CrossEntropyLoss(weight=weights.to(device).float())
        loss = [ce_loss, SoftDiceLoss(global_weight=10.)]
    elif fnmatch(custom_loss, 'wfocal_w*sdice'):
        weights = get_class_weights(dataset_type, signal_type)
        focal_loss = FocalLoss(weight=weights.to(device).float(), global_weight=1., gamma=2.)
        loss = [focal_loss, SoftDiceLoss(global_weight=float(custom_loss[8:-5]))]
    else:
        loss = nn.CrossEntropyLoss()
    return loss


def get_transformations(transform_names, split='train', sizes=None):
    """Create a list of functions used for preprocessing

    PARAMETERS
    ----------
    transform_names: list
        List of str, one for each transformation
    split: str
        Split currently used
    sizes: int or tuple (optional)
        Used for rescaling
        Default: None
    """
    transformations = list()
    if 'rescale' in transform_names:
        transformations.append(Rescale(sizes))
    if 'flip' in transform_names and split == 'train':
        transformations.append(Flip(0.5))
    if 'vflip' in transform_names and split == 'train':
        transformations.append(VFlip())
    if 'hflip' in transform_names and split == 'train':
        transformations.append(HFlip())
    if 'gain' in transform_names and split == 'train':
        transformations.append(GainJitter())
    if 'noise' in transform_names and split == 'train':
        transformations.append(NoiseJitter())
    return transformations


def mask_to_img(mask, num_class):
    """Generate colors per class, only <=7 classes are supported"""
    # red, green, blue, yellow, cyan, white
    colors = [[255, 0, 0],
              [0, 255, 0],
              [0, 0, 255],
              [255, 255, 0],
              [0, 255, 255],
              [255, 255, 255]
              ]
    mask_img = np.zeros((mask.shape[0],
                         mask.shape[1], 3), dtype=np.uint8)
    for i in range(1, num_class):
        mask_img[mask == i] = colors[i-1]
    mask_img = Image.fromarray(mask_img)
    return mask_img


def get_qualitatives(outputs, masks, paths, seq_name, quali_iter, signal_type=None, dataset_type=None):
    """
    Method to get qualitative results

    PARAMETERS
    ----------
    outputs: torch tensor
        Predicted masks
    masks: torch tensor
        Ground truth masks
    paths: dict
    seq_name: str
    quali_iter: int
        Current iteration on the dataset
    signal_type: str

    RETURNS
    -------
    quali_iter: int
    """
    if signal_type:
        folder_path = paths['logs'] / signal_type / seq_name[0]
        mask_fpath = folder_path / 'mask'
        output_fpath = folder_path / 'output'
    else:
        folder_path = paths['logs'] / seq_name[0]
        mask_fpath = folder_path / 'mask'
        output_fpath = folder_path / 'output'
    
    if dataset_type == 'KuRALS_CW':
        num_class = 4
    elif dataset_type == 'KuRALS_PD':
        num_class = 5
    else:
        raise KeyError(f'Dataset {dataset_type} has not been supported yet.')
    
    # folder_path.mkdir(parents=True, exist_ok=True)
    mask_fpath.mkdir(parents=True, exist_ok=True)
    output_fpath.mkdir(parents=True, exist_ok=True)
    outputs = torch.argmax(outputs, axis=1).cpu().numpy()
    masks = torch.argmax(masks, axis=1).cpu().numpy()
    for i in range(outputs.shape[0]):
        mask_img = mask_to_img(masks[i], num_class)
        mask_path = mask_fpath / 'mask_{}.png'.format(quali_iter)
        mask_img.save(mask_path)
        output_img = mask_to_img(outputs[i], num_class)
        output_path = output_fpath / 'output_{}.png'.format(quali_iter)
        output_img.save(output_path)
        quali_iter += 1
    return quali_iter


def count_params(model):
    """Count trainable parameters of a PyTorch Model"""
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    nb_params = sum([np.prod(p.size()) for p in model_parameters])
    return nb_params
