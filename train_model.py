"""Dataset loading and background training for the driving model."""

from __future__ import annotations

import csv
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch
from PyQt6.QtCore import QThread, pyqtSignal
from torch import nn
from torch.utils.data import DataLoader, Dataset

from model import (
    CLASS_NAMES, FORWARD_INPUT, MINIMAP_INPUT, DrivingNet, preprocess_bgr, save_checkpoint,
)

MODELS_DIR = Path(__file__).with_name("models")
DATASET_DIR = Path(__file__).with_name("dataset")

Sample = tuple[Path, Path, int]


def describe_device() -> str:
    if torch.cuda.is_available():
        return f"CUDA available: {torch.cuda.get_device_name(0)}"
    return "CUDA not available - training on CPU"


def find_sessions(root: str | Path = DATASET_DIR) -> list[Path]:
    return sorted(Path(root).glob("*/*/telemetry.csv"))


def read_session(csv_path: Path) -> list[Sample]:
    samples: list[Sample] = []
    base = csv_path.parent
    try:
        with csv_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                try:
                    label = int(row["steer_label"])
                    frame, minimap = base / row["frame_file"], base / row["minimap_file"]
                except (KeyError, ValueError, TypeError):
                    continue
                if 0 <= label < len(CLASS_NAMES) and frame.is_file() and minimap.is_file():
                    samples.append((frame, minimap, label))
    except (OSError, csv.Error):
        pass
    return samples


def split_sessions(
    root: str | Path = DATASET_DIR, val_fraction: float = 0.2
) -> tuple[list[Sample], list[Sample]]:
    """Hold out the tail of every session: adjacent 5 Hz frames are near-duplicates,
    so a random split would leak training frames into validation."""
    train: list[Sample] = []
    val: list[Sample] = []
    for path in find_sessions(root):
        samples = read_session(path)
        cut = len(samples) - int(len(samples) * val_fraction)
        if len(samples) >= 5 and cut == len(samples):
            cut -= 1
        train += samples[:cut]
        val += samples[cut:]
    return train, val


def class_weights(labels: list[int]) -> torch.Tensor:
    """Inverse-frequency weights, normalized to mean 1 over present classes."""
    counts = np.bincount(labels, minlength=len(CLASS_NAMES)).astype(np.float64)
    weights = np.zeros_like(counts)
    present = counts > 0
    weights[present] = counts.sum() / (present.sum() * counts[present])
    return torch.tensor(weights, dtype=torch.float32)


def augment(image: np.ndarray, rng: random.Random) -> np.ndarray:
    """Random contrast/brightness shifts to mimic overcast and low-light weather."""
    alpha = rng.uniform(0.6, 1.2)
    beta = rng.uniform(-35.0, 25.0)
    out = image.astype(np.float32) * alpha + beta
    if rng.random() < 0.5:  # desaturate toward gray like heavy overcast
        gray = out.mean(axis=2, keepdims=True)
        mix = rng.uniform(0.2, 0.6)
        out = out * (1 - mix) + gray * mix
    return np.clip(out, 0, 255).astype(np.uint8)


class DrivingDataset(Dataset):
    def __init__(self, samples: list[Sample], train: bool = False, seed: int = 0):
        self.samples = samples
        self.train = train
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        frame_path, minimap_path, label = self.samples[index]
        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        minimap = cv2.imread(str(minimap_path), cv2.IMREAD_COLOR)
        if frame is None:
            frame = np.zeros((360, 640, 3), np.uint8)
        if minimap is None:
            minimap = np.zeros((256, 256, 3), np.uint8)
        if self.train:
            rng = random.Random(self._rng.random())  # worker-safe per-item stream
            frame, minimap = augment(frame, rng), augment(minimap, rng)
        return (
            preprocess_bgr(frame, FORWARD_INPUT),
            preprocess_bgr(minimap, MINIMAP_INPUT),
            label,
        )


@dataclass
class TrainParams:
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 1e-3
    backbone: str = "convnet"
    dataset_root: str = str(DATASET_DIR)
    output_dir: str = str(MODELS_DIR)
    num_workers: int = 0


class TrainingStopped(Exception):
    pass


def run_training(
    params: TrainParams,
    on_epoch: Callable[[dict], None] = lambda _: None,
    on_progress: Callable[[int], None] = lambda _: None,
    on_log: Callable[[str], None] = lambda _: None,
    should_stop: Callable[[], bool] = lambda: False,
) -> Path:
    """Train and return the best-validation checkpoint path."""
    train_samples, val_samples = split_sessions(params.dataset_root)
    if not train_samples:
        raise RuntimeError("No training samples found. Record a dataset first.")
    if not val_samples:
        val_samples = train_samples[-max(1, len(train_samples) // 5):]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    on_log(f"{describe_device()} | {len(train_samples)} train / {len(val_samples)} val samples")

    train_loader = DataLoader(
        DrivingDataset(train_samples, train=True), batch_size=params.batch_size,
        shuffle=True, num_workers=params.num_workers, drop_last=len(train_samples) > params.batch_size,
    )
    val_loader = DataLoader(
        DrivingDataset(val_samples), batch_size=params.batch_size, num_workers=params.num_workers,
    )
    net = DrivingNet(params.backbone).to(device)
    weights = class_weights([label for _, _, label in train_samples]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(net.parameters(), lr=params.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, params.epochs))
    on_log("Class weights: " + ", ".join(f"{n}={w:.2f}" for n, w in zip(CLASS_NAMES, weights.tolist())))

    out_dir = Path(params.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / f"driver_{time.strftime('%Y%m%d_%H%M%S')}.pt"
    best_acc = -1.0
    total_steps = max(1, params.epochs * len(train_loader))
    step = 0
    for epoch in range(1, params.epochs + 1):
        net.train()
        running, seen = 0.0, 0
        for road, mini, labels in train_loader:
            if should_stop():
                raise TrainingStopped()
            road, mini, labels = road.to(device), mini.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(net(road, mini), labels)
            loss.backward()
            optimizer.step()
            running += loss.item() * labels.size(0)
            seen += labels.size(0)
            step += 1
            on_progress(int(100 * step / total_steps))
        scheduler.step()

        net.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for road, mini, labels in val_loader:
                if should_stop():
                    raise TrainingStopped()
                road, mini, labels = road.to(device), mini.to(device), labels.to(device)
                logits = net(road, mini)
                val_loss += criterion(logits, labels).item() * labels.size(0)
                correct += (logits.argmax(1) == labels).sum().item()
                total += labels.size(0)
        stats = {
            "epoch": epoch,
            "train_loss": running / max(1, seen),
            "val_loss": val_loss / max(1, total),
            "val_acc": 100.0 * correct / max(1, total),
        }
        if stats["val_acc"] > best_acc:
            best_acc = stats["val_acc"]
            save_checkpoint(best_path, net, {
                "epoch": epoch, "val_acc": best_acc, "params": params.__dict__,
            })
        on_epoch(stats)
        on_log(
            f"Epoch {epoch}/{params.epochs}  loss {stats['train_loss']:.4f}  "
            f"val_loss {stats['val_loss']:.4f}  val_acc {stats['val_acc']:.1f}%"
        )
    on_progress(100)
    return best_path


class TrainingThread(QThread):
    progress = pyqtSignal(int)
    epoch_done = pyqtSignal(dict)
    log = pyqtSignal(str)
    finished_ok = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, params: TrainParams, parent=None):
        super().__init__(parent)
        self.params = params
        self._stop_requested = False

    def request_stop(self) -> None:
        self._stop_requested = True

    def run(self) -> None:
        try:
            path = run_training(
                self.params,
                on_epoch=self.epoch_done.emit,
                on_progress=self.progress.emit,
                on_log=self.log.emit,
                should_stop=lambda: self._stop_requested,
            )
        except TrainingStopped:
            self.failed.emit("Training stopped.")
        except Exception as exc:
            self.failed.emit(f"Training failed: {exc}")
        else:
            self.finished_ok.emit(str(path))
