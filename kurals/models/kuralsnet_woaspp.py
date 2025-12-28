import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConvBlock(nn.Module):
    """ (2D conv => BN => LeakyReLU) * 2 """

    def __init__(self, in_ch, out_ch, k_size, pad, dil):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k_size, padding=pad, dilation=dil),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=k_size, padding=pad, dilation=dil),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.block(x)
        return x


class Double3DConvBlock(nn.Module):
    """ (3D conv => BN => LeakyReLU) * 2 """

    def __init__(self, in_ch, out_ch, k_size, pad, dil):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=k_size, padding=pad, dilation=dil),
            nn.BatchNorm3d(out_ch),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=k_size, padding=pad, dilation=dil),
            nn.BatchNorm3d(out_ch),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.block(x)
        return x


class ConvBlock(nn.Module):
    """ (2D conv => BN => LeakyReLU) """

    def __init__(self, in_ch, out_ch, k_size, pad, dil):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k_size, padding=pad, dilation=dil),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.block(x)
        return x


class EncodingBranch(nn.Module):
    """
    Encoding branch for a single radar view

    PARAMETERS
    ----------
    signal_type: str
        Type of radar view.
        Supported: 'range_doppler', 'range_angle' and 'angle_doppler'
    """

    def __init__(self, signal_type, n_frames, dataset_type='KuRALS_CW'):
        super().__init__()
        self.signal_type = signal_type
        self.dataset_type = dataset_type
        self.double_3dconv_block1 = Double3DConvBlock(in_ch=1, out_ch=32, k_size=(1+n_frames//2, 3, 3),
                                                      pad=(0, 1, 1), dil=1)
        self.doppler_max_pool = nn.MaxPool2d(2, stride=(1, 2))
        self.max_pool = nn.MaxPool2d(2, stride=2)

        self.double_conv_block = DoubleConvBlock(in_ch=32, out_ch=64, k_size=3,
                                                pad=1, dil=1)

        self.single_conv_block1_1x1 = ConvBlock(in_ch=64, out_ch=128, k_size=1,
                                                pad=0, dil=1)

    def forward(self, x):
        x1 = self.double_3dconv_block1(x)
        x1 = torch.squeeze(x1, 2)  # remove temporal dimension

        # For KuRALS_CW dataset
        if self.dataset_type == 'KuRALS_CW':
            x1_pad = F.pad(x1, (0, 0, 0, 0), "constant", 0)
            x1_down = self.max_pool(x1_pad)
        # For KuRALS_PD dataset
        else:
            x1_pad = F.pad(x1, (0, 0, 0, 1), "constant", 0)
            x1_down = self.doppler_max_pool(x1_pad)

        x2 = self.double_conv_block(x1_down)
        x2_pad = F.pad(x2, (0, 0, 0, 1), "constant", 0)
        x2_down = self.doppler_max_pool(x2_pad)

        x3 = self.single_conv_block1_1x1(x2_down)
        # return input of ASPP block + latent features
        return x3


class KuRALSNet_WoASPP(nn.Module):
    """
    KuRALS-Net without ASPP module

    PARAMETERS
    ----------
    n_classes: int
        Number of classes used for the semantic segmentation task
    n_frames: int
        Total numer of frames used as a sequence
    """

    def __init__(self, n_classes, n_frames, dataset_type='KuRALS_CW'):
        super().__init__()
        self.n_classes = n_classes
        self.n_frames = n_frames

        # Backbone (encoding)
        self.rd_encoding_branch = EncodingBranch('range_doppler', n_frames, dataset_type)

        # Range-Doppler (RD) decoding branch
        # For KuRALS_CW dataset
        if dataset_type == 'KuRALS_CW':
            self.rd_upconv1 = nn.ConvTranspose2d(128, 128, (2, 2), stride=(2, 2))
        # For KuRALS_PD dataset
        else:
            self.rd_upconv1 = nn.ConvTranspose2d(128, 128, (1, 2), stride=(1, 2))
        self.rd_double_conv_block1 = DoubleConvBlock(in_ch=128, out_ch=128, k_size=3,
                                                     pad=1, dil=1)
        self.rd_upconv2 = nn.ConvTranspose2d(128, 128, (1, 2), stride=(1, 2))
        self.rd_double_conv_block2 = DoubleConvBlock(in_ch=128, out_ch=128, k_size=3,
                                                     pad=1, dil=1)
        
        # Final 1D convs
        self.rd_final = nn.Conv2d(in_channels=128, out_channels=n_classes, kernel_size=1)

    def forward(self, x_rd):
        # Backbone
        rd_latent = self.rd_encoding_branch(x_rd)


        # Decoding branch with upconvs
        x4_rd = self.rd_upconv1(rd_latent)
        x4_rd = self.rd_double_conv_block1(x4_rd)

        x4_rd = self.rd_upconv2(x4_rd)
        x4_rd = self.rd_double_conv_block2(x4_rd)
        
        # Final 1D convolutions
        x4_rd = self.rd_final(x4_rd)

        return x4_rd