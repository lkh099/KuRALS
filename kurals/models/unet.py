# from https://github.com/milesial/Pytorch-UNet/tree/master
import torch
import torch.nn as nn
import torch.nn.functional as F


def pool_plan(input_size, n_down):
    """Per-stage (kh, kw) pooling kernels for an input of `input_size`.

    A stock U-Net halves both axes at every stage, which collapses a short
    axis to zero: an 8x64 RD map dies at down4 (8 -> 4 -> 2 -> 1 -> 0). Here an
    axis is only halved while at least 2 cells survive, so the 8-wide Doppler
    axis stops downsampling once it reaches 1 while range keeps going. With
    input_size=None every kernel is (2, 2) and the net is the stock one.
    """
    if input_size is None:
        return [(2, 2)] * n_down
    h, w = input_size
    plan = []
    for _ in range(n_down):
        kh = 2 if h >= 2 else 1
        kw = 2 if w >= 2 else 1
        h //= kh
        w //= kw
        plan.append((kh, kw))
    return plan


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, pool=(2, 2)):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(pool),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, pool=(2, 2)):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = None
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=pool, stride=pool)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        # Upsample to the skip connection's own size rather than by a fixed
        # factor of 2. That inverts whatever pool_plan did on each axis, so an
        # anisotropic encoder round-trips exactly and the old pad/crop of a
        # one-off size mismatch is no longer needed.
        if self.up is not None:
            x1 = self.up(x1)
        if x1.shape[-2:] != x2.shape[-2:]:
            x1 = F.interpolate(x1, size=x2.shape[-2:], mode='bilinear', align_corners=True)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_classes, n_frames=5, bilinear=True, input_size=None):
        """
        Args:
            n_classes : _description_
            n_frames : _description_. Defaults to 5.
            bilinear : Bilinear interpolation or ConvTranspose2d.
            input_size : (H, W) of the input, used to plan the pooling so a
                short axis is not collapsed. None keeps the stock all-(2,2)
                schedule. Channel widths are unchanged either way, so the
                parameter count stays that of the published U-Net.
        """
        super(UNet, self).__init__()
        self.n_channels = n_frames
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.input_size = input_size

        plan = pool_plan(input_size, 4)
        self.pool_plan = plan

        self.inc = (DoubleConv(n_frames, 64))
        self.down1 = (Down(64, 128, plan[0]))
        self.down2 = (Down(128, 256, plan[1]))
        self.down3 = (Down(256, 512, plan[2]))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor, plan[3]))
        self.up1 = (Up(1024, 512 // factor, bilinear, plan[3]))
        self.up2 = (Up(512, 256 // factor, bilinear, plan[2]))
        self.up3 = (Up(256, 128 // factor, bilinear, plan[1]))
        self.up4 = (Up(128, 64, bilinear, plan[0]))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
