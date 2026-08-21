"""A lot of functions used in our pipelines"""
import json
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
