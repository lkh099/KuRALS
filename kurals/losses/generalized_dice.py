import torch
from torch import nn
from typing import Union
import torch.nn.functional as F
from .one_hot import one_hot


class GeneralizedDiceLoss(nn.Module):
    """
    Compute the generalised Dice loss defined in:

        Sudre, C. et. al. (2017) Generalised Dice overlap as a deep learning
        loss function for highly unbalanced segmentations. DLMIA 2017.

    Adapted from:
        https://github.com/Project-MONAI/MONAI/blob/dev/monai/losses/dice.py
    """

    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = True,
        sigmoid: bool = False,
        softmax: bool = True,
        squared_pred: bool = False,
        reduction: str = 'mean',
        smooth: float = 1.0,
        eps: float = 1e-8,
        batch: bool = False,
        w_type: Union[str, list] = 'square',
    ) -> None:
        """
        Args:
            include_background: If False channel index 0 (background category) is excluded from the calculation.
            to_onehot_y: whether to convert the ``target`` into the one-hot format,
                using the number of classes inferred from `input` (``input.shape[1]``). Defaults to False.
            sigmoid: If True, apply a sigmoid function to the prediction.
            softmax: If True, apply a softmax function to the prediction.
            squared_pred: use squared versions of targets and predictions in the denominator or not.
            reduction: {``"none"``, ``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.

                - ``"none"``: no reduction will be applied.
                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.
            smooth: laplace smoothing to smooth dice loss and accelerate convergence. Default: 1.0
            eps: a small constant added to the denominator to avoid nan. Default: 1e-8
            batch: whether to sum the intersection and union areas over the batch dimension before the dividing.
                Defaults to False, intersection over union is computed from each item in the batch.
                If True, the class-weighted intersection and union areas are first summed across the batches.
            w_type: {``"square"``, ``"simple"``, ``"uniform"``}
                Type of function to transform ground truth volume to a weight factor. Defaults to ``"square"``.

        Raises:
            ValueError: When more than 1 of [``sigmoid=True``, ``softmax=True``].
                Incompatible values.

        """
        super().__init__()
        if int(sigmoid) + int(softmax) > 1:
            raise ValueError("Incompatible values: more than 1 of [sigmoid=True, softmax=True].")
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.squared_pred = squared_pred
        self.reduction = reduction
        self.smooth = float(smooth)
        self.eps = float(eps)
        self.batch = batch
        self.w_type = w_type

    def w_func(self, grnd, pred):
        if self.w_type == 'simple':
            return torch.reciprocal(grnd)
        elif self.w_type == 'square':
            return torch.reciprocal(grnd * grnd)
        elif self.w_type == 'uniform':
            return torch.ones_like(grnd)
        elif self.w_type == 'scale':
            return (torch.max(grnd, pred) + 1.0) / (torch.min(grnd, pred) + 1.0)
        else:
            raise ValueError(f'Unsupported w_type: {self.w_type}, available options are ["square", "simple", "uniform"].')

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD], where N is the number of classes.
            target: the shape should be BNH[WD] or B1H[WD], where N is the number of classes.

        Raises:
            AssertionError: When input and target (after one hot transform if set)
                have different shapes.
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].

        """
        ## Preparing
        if self.sigmoid:
            input = torch.sigmoid(input)
        
        num_class = input.shape[1]
        if self.softmax:
            if num_class == 1:
                raise Warning("single channel prediction, `softmax=True` ignored.")
            else:
                input = F.softmax(input, dim=1)

        if self.to_onehot_y:
            if num_class == 1:
                raise Warning("single channel prediction, `to_onehot_y=True` ignored.")
            else:
                target = one_hot(labels=target, num_classes=num_class, device=target.device, dtype=target.dtype)

        if not self.include_background:
            if num_class == 1:
                raise Warning("single channel prediction, `include_background=False` ignored.")
            else:
                # if skipping background, removing first channel
                target = target[:, 1:]
                input = input[:, 1:]

        if target.shape != input.shape:
            raise AssertionError(f"ground truth has differing shape ({target.shape}) from input ({input.shape})")

        ## Calculating Dice loss
        
        # reducing only spatial dimensions (not batch nor channels)
        reduce_axis: list[int] = torch.arange(2, len(input.shape)).tolist()
        if self.batch:
            reduce_axis = [0] + reduce_axis
        intersection = torch.sum(target * input, reduce_axis)

        if self.squared_pred:
            ground_o = torch.sum(target**2, dim=reduce_axis)
            pred_o = torch.sum(input**2, dim=reduce_axis)
        else:
            ground_o = torch.sum(target, dim=reduce_axis)
            pred_o = torch.sum(input, dim=reduce_axis)

        denominator = ground_o + pred_o

        w = self.w_func(torch.sum(target, reduce_axis).float(), torch.sum(input, dim=reduce_axis).float()) # shape as ground_o
        infs = torch.isinf(w)
        if self.batch:
            # clamp inf to the maximum value
            w[infs] = 0.0
            w = w + infs * torch.max(w)
        else:
            # clamp inf to the maximum value per-instance
            w[infs] = 0.0
            max_values = torch.max(w, dim=1)[0].unsqueeze(dim=1)
            w = w + infs * max_values

        final_reduce_dim = 0 if self.batch else 1
        numer = 2.0 * (intersection * w).sum(final_reduce_dim, keepdim=True) + self.smooth
        denom = (denominator * w).sum(final_reduce_dim, keepdim=True) + self.smooth + self.eps
        f: torch.Tensor = 1.0 - (numer / denom)

        if self.reduction == 'mean':
            f = torch.mean(f)  # the batch and channel average
        elif self.reduction == 'sum':
            f = torch.sum(f)  # sum over the batch and channel dims
        elif self.reduction == 'none':
            # If we are not computing voxelwise loss components at least
            # make sure a none reduction maintains a broadcastable shape
            broadcast_shape = list(f.shape[0:2]) + [1] * (len(input.shape) - 2)
            f = f.view(broadcast_shape)
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')

        return f

if __name__ == "__main__":
    torch.manual_seed(42)

    pred = torch.rand(2, 3, 4, 4, requires_grad=True)
    # print('prediction: ', pred)
    target = torch.randint(low=0, high=3, size=(2, 4, 4))
    # print('targets: ', target)
    dice_loss = GeneralizedDiceLoss(squared_pred=True, batch=False, w_type='square', smooth=1e-6)

    loss = dice_loss(pred, target)
    print('loss: ', loss)
    loss.backward()
    print(pred.grad)