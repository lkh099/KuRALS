"""Class to computes metrics for Carrada"""
import numpy as np
from scipy.stats import hmean
# from sklearn.metrics import confusion_matrix
import torch
from kurals.utils.distributed_utils import reduce_value

# from https://stackoverflow.com/questions/59080843/faster-method-of-computing-confusion-matrix
def confusion_matrix(y_true, y_pred, num_class):
    N = num_class
    y = N * y_true + y_pred
    y = torch.bincount(y)
    if len(y) < N * N:
        y = torch.cat((y, torch.zeros(N * N - len(y), dtype=y.dtype, device=y.device)))
    y = y.reshape(N, N)
    return y

class Evaluator:
    """Class to evaluate a model with quantitative metrics
    using a ground truth mask and a predicted mask.

    PARAMETERS
    ----------
    num_class: int
    """

    def __init__(self, num_class):
        self.num_class = num_class
        # num_gt * num_pred
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def get_pixel_prec_class(self, harmonic_mean=False):
        """Pixel Precision"""
        # summation according to the column direction, i.e., \sum_{j=0}^{n}p_{ji}
        prec_by_class = np.diag(self.confusion_matrix) / np.nansum(self.confusion_matrix, axis=0)
        prec_by_class = np.nan_to_num(prec_by_class)
        if harmonic_mean:
            prec = hmean(prec_by_class)
        else:
            prec = np.mean(prec_by_class)
        return prec, prec_by_class

    def get_pixel_recall_class(self, harmonic_mean=False):
        """Pixel Recall"""
        # pixel recall equals to detection probability 
        # summation according to the row direction, i.e., \sum_{j=0}^{n}p_{ij}
        recall_by_class = np.diag(self.confusion_matrix) / np.nansum(self.confusion_matrix, axis=1)
        recall_by_class = np.nan_to_num(recall_by_class)
        if harmonic_mean:
            recall = hmean(recall_by_class)
        else:
            recall = np.mean(recall_by_class)
        return recall, recall_by_class

    def get_pixel_acc_class(self, harmonic_mean=False):
        """(deprecated) Pixel Accuracy"""
        acc_by_class = np.diag(self.confusion_matrix).sum() / (np.nansum(self.confusion_matrix, axis=1)
                                                               + np.nansum(self.confusion_matrix, axis=0)
                                                               + np.diag(self.confusion_matrix).sum()
                                                               - 2*np.diag(self.confusion_matrix))
        acc_by_class = np.nan_to_num(acc_by_class)
        if harmonic_mean:
            acc = hmean(acc_by_class)
        else:
            acc = np.mean(acc_by_class)
        return acc, acc_by_class
    
    def get_pixel_far_class(self, harmonic_mean=False):
        """Pixel False Alarm Rate"""
        far_by_class = (np.nansum(self.confusion_matrix, axis=0) - np.diag(self.confusion_matrix)) / (
                            np.sum(self.confusion_matrix) - np.nansum(self.confusion_matrix, axis=1)
                            )
        far_by_class = np.nan_to_num(far_by_class)
        if harmonic_mean:
            prec = hmean(far_by_class)
        else:
            prec = np.mean(far_by_class)
        return prec, far_by_class
    
    def get_pixel_acc(self):
        """Pixel Accuracy"""
        acc = np.diag(self.confusion_matrix).sum() / np.nansum(self.confusion_matrix, axis=(0,1))
        acc = np.nan_to_num(acc)
        return acc

    def get_miou_class(self, harmonic_mean=False):
        """Mean Intersection over Union"""
        miou_by_class = np.diag(self.confusion_matrix) / (np.nansum(self.confusion_matrix, axis=1)
                                                          + np.nansum(self.confusion_matrix, axis=0)
                                                          - np.diag(self.confusion_matrix))
        miou_by_class = np.nan_to_num(miou_by_class)
        if harmonic_mean:
            miou = hmean(miou_by_class)
        else:
            miou = np.mean(miou_by_class)
        return miou, miou_by_class

    def get_dice_class(self, harmonic_mean=False):
        """Dice or F1 score"""
        _, prec_by_class = self.get_pixel_prec_class()
        _, recall_by_class = self.get_pixel_recall_class()
        # Add epsilon term to avoid /0
        dice_by_class = 2*prec_by_class*recall_by_class/(prec_by_class + recall_by_class + 1e-8)
        if harmonic_mean:
            dice = hmean(dice_by_class)
        else:
            dice = np.mean(dice_by_class)
        return dice, dice_by_class
    
    # def _generate_matrix(self, labels, predictions):
    #     matrix = confusion_matrix(labels.flatten(), predictions.flatten(),
    #                               labels=list(range(self.num_class)))
    #     return matrix
    
    def _generate_matrix(self, labels, predictions):
        matrix = confusion_matrix(labels.flatten(), predictions.flatten(),
                                  num_class=self.num_class)
        return matrix.cpu().numpy()

    def add_batch(self, labels, predictions):
        """Method to add ground truth and predicted masks by batch
        and update the global confusion matrix (entire dataset)

        PARAMETERS
        ----------
        labels: torch tensor or numpy array
            Ground truth masks
        predictions: torch tensor or numpy array
            Predicted masks
        """
        assert labels.shape == predictions.shape
        self.confusion_matrix += self._generate_matrix(labels, predictions)

    def reset(self):
        """Method to reset the confusion matrix"""
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def update(self, confusion_matrix):
        self.confusion_matrix = confusion_matrix.numpy()
    
    def get(self):
        return torch.from_numpy(self.confusion_matrix)
    
    # def add_batch_dist(self, labels, predictions):
    #     assert labels.shape == predictions.shape
    #     conf_matrix = torch.from_numpy(self._generate_matrix(labels.cpu(), predictions.cpu())).to('cuda')
    #     conf_matrix = reduce_value(conf_matrix, average=False)
    #     self.confusion_matrix += conf_matrix.cpu().numpy()
    
    def add_batch_dist(self, labels, predictions):
        assert labels.shape == predictions.shape
        conf_matrix = confusion_matrix(labels.flatten(), predictions.flatten(),
                                  num_class=self.num_class)
        conf_matrix = reduce_value(conf_matrix, average=False)
        self.confusion_matrix += conf_matrix.cpu().numpy()
