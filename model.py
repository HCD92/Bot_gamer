"""Behavioral-cloning network: forward road view + minimap -> steering class logits."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

CLASS_NAMES = ("STRAIGHT", "LEFT", "RIGHT", "BRAKE")
NUM_CLASSES = len(CLASS_NAMES)
FORWARD_INPUT = (3, 128, 224)  # C, H, W fed to the network
MINIMAP_INPUT = (3, 96, 96)
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def _conv_stack(channels: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for cin, cout in zip(channels[:-1], channels[1:]):
        layers += [
            nn.Conv2d(cin, cout, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        ]
    layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten()]
    return nn.Sequential(*layers)


def _resnet18_trunk() -> tuple[nn.Module, int]:
    from torchvision.models import resnet18

    net = resnet18(weights=None)  # trained from scratch on the recorded data
    net.fc = nn.Identity()
    return net, 512


class DrivingNet(nn.Module):
    """Two-branch CNN. ``backbone`` is "convnet" (4 conv layers) or "resnet18" for the road view."""

    def __init__(self, backbone: str = "convnet", num_classes: int = NUM_CLASSES, dropout: float = 0.3):
        super().__init__()
        self.backbone = backbone
        if backbone == "resnet18":
            self.road, road_features = _resnet18_trunk()
        elif backbone == "convnet":
            self.road, road_features = _conv_stack([3, 24, 48, 96, 128]), 128
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        self.minimap = _conv_stack([3, 16, 32, 48, 64])
        self.head = nn.Sequential(
            nn.Linear(road_features + 64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, road: torch.Tensor, minimap: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([self.road(road), self.minimap(minimap)], dim=1))


def preprocess_bgr(image: np.ndarray, size: tuple[int, int, int]) -> torch.Tensor:
    """BGR uint8 HxWx3 -> normalized float CHW tensor at ``size`` (C, H, W)."""
    import cv2

    _, height, width = size
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    rgb = resized[:, :, ::-1].astype(np.float32) / 255.0
    rgb = (rgb - np.asarray(MEAN, dtype=np.float32)) / np.asarray(STD, dtype=np.float32)
    return torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))


def save_checkpoint(path: str | Path, model: DrivingNet, meta: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "backbone": model.backbone,
        "classes": list(CLASS_NAMES),
        "meta": meta,
    }, path)


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> DrivingNet:
    data = torch.load(path, map_location=device, weights_only=True)
    model = DrivingNet(data.get("backbone", "convnet"))
    model.load_state_dict(data["state_dict"])
    return model.to(device).eval()


class DrivingPolicy:
    """Inference wrapper used by the agent for autonomous driving."""

    def __init__(self, path: str | Path, device: str | None = None):
        self.path = str(path)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = load_checkpoint(path, self.device)

    @torch.no_grad()
    def predict(self, forward_bgr: np.ndarray, minimap_bgr: np.ndarray) -> tuple[int, list[float]]:
        road = preprocess_bgr(forward_bgr, FORWARD_INPUT).unsqueeze(0).to(self.device)
        mini = preprocess_bgr(minimap_bgr, MINIMAP_INPUT).unsqueeze(0).to(self.device)
        probs = torch.softmax(self.model(road, mini), dim=1)[0]
        return int(probs.argmax()), probs.cpu().tolist()
