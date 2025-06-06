import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models.segmentation import deeplabv3_resnet50

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
        )
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.block(x) + self.skip(x))

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.resblock = ResidualBlock(in_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.upsample(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        return self.resblock(torch.cat([x, skip], dim=1))

class UNetResidual(nn.Module):
    def __init__(self, in_channels=4, out_channels=1):
        super().__init__()
        self.enc1 = ResidualBlock(in_channels, 64)
        self.enc2 = ResidualBlock(64, 128)
        self.enc3 = ResidualBlock(128, 256)
        self.enc4 = ResidualBlock(256, 512)

        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ResidualBlock(512, 512)

        self.dec4 = DecoderBlock(512, 512, 256)
        self.dec3 = DecoderBlock(256, 256, 128)
        self.dec2 = DecoderBlock(128, 128, 64)
        self.dec1 = DecoderBlock(64, 64, 64)

        self.dropout = nn.Dropout(p=0.3)
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x, apply_sigmoid=False):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.dropout(self.dec4(b, e4))
        d3 = self.dropout(self.dec3(d4, e3))
        d2 = self.dropout(self.dec2(d3, e2))
        d1 = self.dropout(self.dec1(d2, e1))

        out = self.final(d1)
        return {'out': torch.sigmoid(out) if apply_sigmoid else out}

def build_model():
    return UNetResidual(in_channels=4)


def prebuild_model():
    model = deeplabv3_resnet50(pretrained=False, pretrained_backbone=False)
    model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=1)
    return model
