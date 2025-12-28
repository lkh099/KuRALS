import torch.nn as nn
import torch
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """ (2D conv => BN => ReLU) """

    def __init__(self, in_ch, out_ch, k_size, pad, dil):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k_size, padding=pad, dilation=dil),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.block(x)
        return x

class DoubleConvBlock(nn.Module):
    """ (2D conv => BN => ReLU) * 2"""

    def __init__(self, in_ch, out_ch, k_size, pad, dil):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k_size, padding=pad, dilation=dil),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=k_size, padding=pad, dilation=dil),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.block(x)
        return x

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_ch, out_ch, k_size=3, pad=1, dil=1):
        super().__init__()
        self.conv_maxpool = nn.Sequential(
            DoubleConvBlock(in_ch, out_ch, k_size, pad, dil),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.conv_maxpool(x)

class ASPPBlock(nn.Module):
    """Atrous Spatial Pyramid Pooling
    Parallel conv blocks with different dilation rate
    """

    def __init__(self, in_ch, out_ch=256):
        super().__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        self.conv1_1x1 = nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0, dilation=1)
        self.single_conv_block1_1x1 = ConvBlock(in_ch, out_ch, k_size=1, pad=0, dil=1)
        self.single_conv_block1_3x3 = ConvBlock(in_ch, out_ch, k_size=3, pad=6, dil=6)
        self.single_conv_block2_3x3 = ConvBlock(in_ch, out_ch, k_size=3, pad=12, dil=12)
        self.single_conv_block3_3x3 = ConvBlock(in_ch, out_ch, k_size=3, pad=18, dil=18)

    def forward(self, x):
        b, c, h, w = x.shape
        x1 = F.interpolate(self.global_avg_pool(x), size=(h, w), align_corners=False,
                           mode='bilinear')
        x1 = self.conv1_1x1(x1)
        x2 = self.single_conv_block1_1x1(x)
        x3 = self.single_conv_block1_3x3(x)
        x4 = self.single_conv_block2_3x3(x)
        x5 = self.single_conv_block3_3x3(x)
        x_cat = torch.cat((x2, x3, x4, x5, x1), 1)
        return x_cat

class RSSNet(nn.Module):
    def __init__(self, n_classes, n_frames):
        super().__init__()
        self.n_classes = n_classes
        self.n_frames = n_frames

        # Encoding
        self.down1 = (Down(self.n_frames, 48, dil=2, pad=2))
        self.down2 = (Down(48, 128, dil=2, pad=2))
        self.down3 = (Down(128, 256, dil=2, pad=2))
        self.down4 = (Down(256, 512, dil=2, pad=2))
        
        # ASPP Blocks
        self.aspp_block = ASPPBlock(in_ch=512, out_ch=256)
        self.single_conv_block1_1x1 = ConvBlock(in_ch=1280, out_ch=256, k_size=1, pad=0, dil=1)

        # Decoding
        self.single_conv_block2_1x1 = ConvBlock(in_ch=128, out_ch=48, k_size=1, pad=0, dil=1)
        
        self.double_conv_block = DoubleConvBlock(in_ch=304, out_ch=256, k_size=3, pad=1, dil=1)

        self.single_conv = nn.Conv2d(in_channels=256, out_channels=self.n_classes, kernel_size=3, padding=1)
        self.up = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

    def forward(self, x):
        x1 = self.down1(x) # 1/2
        x2 = self.down2(x1) # 1/4
        x3 = self.down3(x2) # 1/8
        x4 = self.down4(x3) # 1/16
        
        x4_aspp = self.aspp_block(x4)
        x4_aspp = self.single_conv_block1_1x1(x4_aspp)
        x4_aspp = F.interpolate(x4_aspp, size=x2.shape[-2:], align_corners=True,
                           mode='bilinear')
        
        x2_feature = self.single_conv_block2_1x1(x2)
        
        x5 = self.double_conv_block(torch.cat([x2_feature, x4_aspp], dim=1))
        
        x6 = self.single_conv(x5)
        x6_up = self.up(x6)

        return x6_up

if __name__ == '__main__':
    net = RSSNet(n_classes=4, n_frames=1)
    device = torch.device('cpu')
    net.to(device)
    input = torch.randn(size=(1, 1, 124, 2048)).to(device)
    output = net(input)
    print(output.shape)
    