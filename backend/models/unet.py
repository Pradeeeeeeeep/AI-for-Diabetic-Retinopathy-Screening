"""
unet.py — Standard U-Net Architecture for Retinal Vessel Segmentation
Trained on: DRIVE + CHASE_DB1 + STARE datasets
Input: Green channel (1 x 512 x 512)
Output: Binary vessel mask (1 x 512 x 512)
"""

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class _DoubleConv(nn.Module):
        """(Conv → BN → ReLU) × 2 — the basic U-Net building block."""

        def __init__(self, in_ch: int, out_ch: int):
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return self.block(x)


    class _Down(nn.Module):
        """MaxPool → DoubleConv."""

        def __init__(self, in_ch: int, out_ch: int):
            super().__init__()
            self.block = nn.Sequential(
                nn.MaxPool2d(2),
                _DoubleConv(in_ch, out_ch),
            )

        def forward(self, x):
            return self.block(x)


    class _Up(nn.Module):
        """Bilinear upsampling + skip connection + DoubleConv."""

        def __init__(self, in_ch: int, out_ch: int, bilinear: bool = True):
            super().__init__()
            if bilinear:
                self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
                self.conv = _DoubleConv(in_ch, out_ch)
            else:
                self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, kernel_size=2, stride=2)
                self.conv = _DoubleConv(in_ch, out_ch)

        def forward(self, x1, x2):
            x1 = self.up(x1)
            # Pad to match skip connection dimensions
            diff_y = x2.size(2) - x1.size(2)
            diff_x = x2.size(3) - x1.size(3)
            x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                             diff_y // 2, diff_y - diff_y // 2])
            x = torch.cat([x2, x1], dim=1)
            return self.conv(x)


    class UNet(nn.Module):
        """
        Standard U-Net with 4 encoder/decoder levels.

        Architecture:
          Encoder: 64 → 128 → 256 → 512 → 1024
          Decoder: 1024 → 512 → 256 → 128 → 64 → out_channels
          Skip connections at every level.
        """

        def __init__(self, in_channels: int = 1, out_channels: int = 1, bilinear: bool = True):
            super().__init__()
            self.inc = _DoubleConv(in_channels, 64)
            self.down1 = _Down(64, 128)
            self.down2 = _Down(128, 256)
            self.down3 = _Down(256, 512)
            factor = 2 if bilinear else 1
            self.down4 = _Down(512, 1024 // factor)

            self.up1 = _Up(1024, 512 // factor, bilinear)
            self.up2 = _Up(512, 256 // factor, bilinear)
            self.up3 = _Up(256, 128 // factor, bilinear)
            self.up4 = _Up(128, 64, bilinear)

            self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

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
            return self.outc(x)

except ImportError:
    # PyTorch not available — stub for import safety
    class UNet:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch is required for UNet. Install it with: pip install torch")
