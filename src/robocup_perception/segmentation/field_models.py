"""Field architecture implementations retained by the expanded-data study. No pretrained download."""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models

class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1, padding: int | None = None, groups: int = 1):
        super().__init__()
        if padding is None:
            padding = kernel // 2
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class DSConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            ConvBNAct(in_ch, in_ch, 3, stride, groups=in_ch),
            ConvBNAct(in_ch, out_ch, 1, 1, 0),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class FastSCNN(nn.Module):
    def __init__(self, classes: int = 1):
        super().__init__()
        self.down = nn.Sequential(ConvBNAct(3, 32, 3, 2), DSConv(32, 48, 2), DSConv(48, 64, 2))
        self.global_feat = nn.Sequential(DSConv(64, 96, 2), DSConv(96, 128), DSConv(128, 128))
        self.fuse_low = ConvBNAct(64, 128, 1, 1, 0)
        self.fuse = ConvBNAct(128, 128)
        self.head = nn.Sequential(DSConv(128, 128), nn.Dropout(0.1), nn.Conv2d(128, classes, 1))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        size = value.shape[-2:]
        low = self.down(value)
        high = self.global_feat(low)
        high = F.interpolate(high, size=low.shape[-2:], mode="bilinear", align_corners=False)
        fused = self.fuse(high + self.fuse_low(low))
        return F.interpolate(self.head(fused), size=size, mode="bilinear", align_corners=False)


class CompactMobileNetUNet(nn.Module):
    def __init__(self, variant: str, classes: int = 1, probe_size: int = 320):
        super().__init__()
        backbone = models.mobilenet_v3_large(weights=None) if variant == "large" else models.mobilenet_v3_small(weights=None)
        self.encoder = backbone.features
        self.channels = self._infer_channels(probe_size)
        reversed_channels = list(reversed(self.channels))
        self.up_blocks = nn.ModuleList()
        in_ch = reversed_channels[0]
        for skip_ch in reversed_channels[1:]:
            out_ch = min(128, max(32, skip_ch))
            self.up_blocks.append(
                nn.Sequential(ConvBNAct(in_ch + skip_ch, out_ch), ConvBNAct(out_ch, out_ch))
            )
            in_ch = out_ch
        self.head = nn.Conv2d(in_ch, classes, 1)

    def _features(self, value: torch.Tensor) -> list[torch.Tensor]:
        features = []
        last_hw = None
        for layer in self.encoder:
            value = layer(value)
            if value.shape[-2:] != last_hw:
                features.append(value)
                last_hw = value.shape[-2:]
        return features

    def _infer_channels(self, size: int) -> list[int]:
        training = self.encoder.training
        self.encoder.eval()
        with torch.no_grad():
            result = [int(feature.shape[1]) for feature in self._features(torch.zeros(1, 3, size, size))]
        self.encoder.train(training)
        return result

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        size = value.shape[-2:]
        features = self._features(value)
        decoded = features[-1]
        for block, skip in zip(self.up_blocks, reversed(features[:-1])):
            decoded = F.interpolate(decoded, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            decoded = block(torch.cat([decoded, skip], dim=1))
        return F.interpolate(self.head(decoded), size=size, mode="bilinear", align_corners=False)


class MobileNetLRASPP(nn.Module):
    def __init__(self, classes: int = 1, probe_size: int = 320):
        super().__init__()
        self.encoder = models.mobilenet_v3_large(weights=None).features
        channels = self._infer_channels(probe_size)
        low_ch, high_ch = channels[1], channels[-1]
        self.low = ConvBNAct(low_ch, 48, 1, 1, 0)
        self.high = ConvBNAct(high_ch, 128, 1, 1, 0)
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), ConvBNAct(high_ch, 128, 1, 1, 0))
        self.fuse = nn.Sequential(ConvBNAct(176, 128), nn.Conv2d(128, classes, 1))

    def _features(self, value: torch.Tensor) -> list[torch.Tensor]:
        features = []
        last_hw = None
        for layer in self.encoder:
            value = layer(value)
            if value.shape[-2:] != last_hw:
                features.append(value)
                last_hw = value.shape[-2:]
        return features

    def _infer_channels(self, size: int) -> list[int]:
        training = self.encoder.training
        self.encoder.eval()
        with torch.no_grad():
            result = [int(feature.shape[1]) for feature in self._features(torch.zeros(1, 3, size, size))]
        self.encoder.train(training)
        return result

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        size = value.shape[-2:]
        features = self._features(value)
        low, high = features[1], features[-1]
        pooled = self.high(high) + F.interpolate(
            self.pool(high), size=high.shape[-2:], mode="bilinear", align_corners=False
        )
        pooled = F.interpolate(pooled, size=low.shape[-2:], mode="bilinear", align_corners=False)
        output = self.fuse(torch.cat([self.low(low), pooled], dim=1))
        return F.interpolate(output, size=size, mode="bilinear", align_corners=False)


def make_model(architecture: str, classes: int = 1, size: int = 320) -> nn.Module:
    if architecture == "fast_scnn":
        return FastSCNN(classes)
    if architecture == "mobilenetv3small_unet":
        return CompactMobileNetUNet("small", classes, size)
    if architecture == "compact_mobilenetv3large_unet":
        return CompactMobileNetUNet("large", classes, size)
    if architecture == "mobilenetv3large_lraspp":
        return MobileNetLRASPP(classes, size)
    raise KeyError(architecture)
