import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

from .decode_head import BaseDecodeHead
from .psp_head import PPM

def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    if isinstance(size, torch.Size):
        size = tuple(int(x) for x in size)
    return F.interpolate(input, size, scale_factor, mode, align_corners)

class ConvBlock(nn.Module):
    """ (2D conv => BN => LeakyReLU) """

    def __init__(self, in_ch, out_ch, k_size, pad, dil, inplace=True):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k_size, padding=pad, dilation=dil),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=inplace)
        )

    def forward(self, x):
        out = self.block(x)
        return out

class UPerHead(BaseDecodeHead):
    """Unified Perceptual Parsing for Scene Understanding.

    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """

    def __init__(self, in_channels=[128, 256, 512, 1024], in_index=[0, 1, 2, 3], pool_scales=(1, 2, 3, 6), channels=512,
                 dropout_ratio=0.1, num_classes=3, norm_cfg=dict(type='BN', requires_grad=True), align_corners=False,
                 loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)):
        super(UPerHead, self).__init__()
        self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)
        # self._init_inputs(in_channels, in_index, input_transform)
        self.upsample4 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.in_channels = in_channels
        self.channels = channels
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.input_transform = 'multiple_select'
        self.norm_cfg = norm_cfg
        # self.criteria = nn.CrossEntropyLoss()
        self.in_index = in_index
        # self.loss_decode = build_loss(loss_decode)

        self.align_corners = align_corners

        # PSP Module
        self.psp_modules = PPM(
            pool_scales,
            self.in_channels[-1],
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners)
        self.bottleneck = ConvBlock(
                      in_ch=self.in_channels[-1] + len(pool_scales) * self.channels,
                      out_ch=self.channels,
                      k_size=3,
                      pad=1,
                      dil=1)
        # self.bottleneck = ConvModule(
        #     self.in_channels[-1] + len(pool_scales) * self.channels,
        #     self.channels,
        #     3,
        #     padding=1,
        #     conv_cfg=self.conv_cfg,
        #     norm_cfg=self.norm_cfg,
        #     act_cfg=self.act_cfg)
        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            l_conv = ConvBlock(
                      in_ch=in_channels,
                      out_ch=self.channels,
                      k_size=1,
                      pad=0,
                      dil=1,
                      inplace=False)
            # l_conv = ConvModule(
            #     in_channels,
            #     self.channels,
            #     1,
            #     conv_cfg=self.conv_cfg,
            #     norm_cfg=self.norm_cfg,
            #     act_cfg=self.act_cfg,
            #     inplace=False)
            fpn_conv = ConvBlock(
                      in_ch=self.channels,
                      out_ch=self.channels,
                      k_size=3,
                      pad=1,
                      dil=1,
                      inplace=False)
            # fpn_conv = ConvModule(
            #     self.channels,
            #     self.channels,
            #     3,
            #     padding=1,
            #     conv_cfg=self.conv_cfg,
            #     norm_cfg=self.norm_cfg,
            #     act_cfg=self.act_cfg,
            #     inplace=False)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.fpn_bottleneck = ConvBlock(
                      in_ch=len(self.in_channels) * self.channels,
                      out_ch=self.channels,
                      k_size=3,
                      pad=1,
                      dil=1)
        # self.fpn_bottleneck = ConvModule(
        #     len(self.in_channels) * self.channels,
        #     self.channels,
        #     3,
        #     padding=1,
        #     conv_cfg=self.conv_cfg,
        #     norm_cfg=self.norm_cfg,
        #     act_cfg=self.act_cfg)

    def psp_forward(self, inputs):
        """Forward function of PSP module."""
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)

        return output

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                # F.interpolate(input=x, size=inputs[0].shape[2:], align_corners=self.align_corners, mode='bilinear') for x in inputs
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':

            inputs = [inputs[i] for i in self.in_index]
        else:

            inputs = inputs[self.in_index]

        return inputs
    
    def forward(self, inputs):
        """Forward function."""

        inputs = self._transform_inputs(inputs)

        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        laterals.append(self.psp_forward(inputs))

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            # laterals[i - 1] += F.interpolate(input=laterals[i], size=prev_shape, align_corners=self.align_corners, mode='bilinear')
            # fix the bug of inplace: change += to = +
            laterals[i - 1] = laterals[i - 1] + resize(
                laterals[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=self.align_corners)

        # build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels - 1)
        ]
        # append psp feature
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = resize(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        fpn_outs = torch.cat(fpn_outs, dim=1)
        output1 = self.fpn_bottleneck(fpn_outs)
        output = self.cls_seg(output1)
        # return output, output1
        return output


    def cls_seg(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output


    def losses(self, seg_logit, seg_label):
        """Compute segmentation loss."""
        # print('seg_logit.shape', seg_logit.shape)
        # print('seg_label.shape', seg_label.shape)
        loss = dict()
        seg_logit = resize(
            input=seg_logit,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)

        seg_label = seg_label.squeeze(1)
        # loss['loss_seg'] = self.loss_decode(
        #     seg_logit,
        #     seg_label,
        #     ignore_index=self.ignore_index)
        # loss['loss_seg'] = self.criteria(seg_logit, seg_label, ignore_index=self.ignore_index)
        loss['loss_seg'] = torch.nn.functional.cross_entropy(seg_logit, seg_label, ignore_index=self.ignore_index)
        # loss['acc_seg'] = accuracy(seg_logit, seg_label)
        # print('type(loss[loss_seg])',type(loss['loss_seg']))
        return loss, seg_logit


    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_logits, feature = self.forward(inputs)
        feature = self.upsample4(feature)
        losses, inference = self.losses(seg_logits, gt_semantic_seg)
        return losses, inference, feature

    def forward_test(self, inputs, img_metas, train_cfg):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_logits, feature = self.forward(inputs)
        feature = self.upsample4(feature)

        seg_logits= resize(input=seg_logits, size=[256,256], mode='bilinear', align_corners=self.align_corners)
        # losses, inference = self.losses(seg_logits, gt_semantic_seg)
        return seg_logits, feature