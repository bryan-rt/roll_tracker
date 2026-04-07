"""Training wrapper — YOLO fine-tuning with progressive unfreezing."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from loguru import logger


@dataclass
class HyperparameterConfig:
    """Auto-selected hyperparameters based on round and dataset size."""

    freeze: int = 10
    epochs: int = 100
    lr0: float = 0.001
    reason: str = ""


@dataclass
class TrainResult:
    """Results from a training run."""

    model_path: str = ""
    mAP50: float = 0.0
    mAP50_95: float = 0.0
    val_loss: float = 0.0
    training_time_s: float = 0.0
    round_num: int = 0
    total_frames: int = 0
    hyperparameters: Dict[str, Any] = field(default_factory=dict)


def suggest_hyperparameters(
    round_num: int,
    total_frames: int,
) -> HyperparameterConfig:
    """Select hyperparameters based on round number and dataset size.

    Progressive unfreezing strategy:
    - Round 1 (< 5K): freeze=10 (backbone frozen), epochs=100, lr=0.001
    - Round 2 (5-15K): freeze=6 (partial backbone), epochs=80, lr=0.0005
    - Round 3 (15-50K): freeze=3 (most unfrozen), epochs=60, lr=0.0003
    - Round 4+ (50K+): freeze=0 (full fine-tune), epochs=50, lr=0.0001
    """
    if round_num <= 1 or total_frames < 5000:
        return HyperparameterConfig(
            freeze=10, epochs=100, lr0=0.001,
            reason=(
                f"Round {round_num} with {total_frames} frames: backbone frozen "
                "(head only). Small dataset — prevent overfitting."
            ),
        )
    elif total_frames < 15000:
        return HyperparameterConfig(
            freeze=6, epochs=80, lr0=0.0005,
            reason=(
                f"Round {round_num} with {total_frames} frames: partial backbone "
                "unfreezing. Dataset large enough for some backbone adaptation."
            ),
        )
    elif total_frames < 50000:
        return HyperparameterConfig(
            freeze=3, epochs=60, lr0=0.0003,
            reason=(
                f"Round {round_num} with {total_frames} frames: most layers unfrozen. "
                "Substantial dataset supports deeper fine-tuning."
            ),
        )
    else:
        return HyperparameterConfig(
            freeze=0, epochs=50, lr0=0.0001,
            reason=(
                f"Round {round_num} with {total_frames} frames: full fine-tune. "
                "Large dataset — all layers trainable with low LR."
            ),
        )


def train_model(
    dataset_yaml: str | Path,
    base_model: str | Path,
    round_num: int,
    total_frames: int,
    output_dir: Optional[Path] = None,
    overrides: Optional[Dict[str, Any]] = None,
    device: str = "mps",
    imgsz: int = 640,
    batch_size: int = 8,
) -> TrainResult:
    """Fine-tune a YOLO pose model for one training round.

    Parameters
    ----------
    dataset_yaml : Path to ultralytics dataset.yaml.
    base_model : Path to the base model (.pt) to fine-tune.
    round_num : Current training round number.
    total_frames : Total frames in the dataset (for hyperparameter selection).
    output_dir : Where to save results. Defaults to models/training_runs/round_{N}/.
    overrides : Optional dict of hyperparameter overrides (freeze, epochs, lr0).
    device : Training device (mps, cuda, cpu).
    imgsz : Image size for training.
    batch_size : Training batch size.
    """
    import time

    from ultralytics import YOLO

    output_dir = output_dir or Path(f"models/training_runs/round_{round_num}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Auto-select hyperparameters
    hp = suggest_hyperparameters(round_num, total_frames)
    logger.info(f"Suggested hyperparameters: {hp.reason}")

    # Apply overrides
    freeze = hp.freeze
    epochs = hp.epochs
    lr0 = hp.lr0
    if overrides:
        freeze = overrides.get("freeze", freeze)
        epochs = overrides.get("epochs", epochs)
        lr0 = overrides.get("lr0", lr0)
        if overrides:
            logger.info(f"Applied overrides: {overrides}")

    # Save training args for reproducibility
    train_args = {
        "base_model": str(base_model),
        "dataset_yaml": str(dataset_yaml),
        "round": round_num,
        "total_frames": total_frames,
        "freeze": freeze,
        "epochs": epochs,
        "lr0": lr0,
        "device": device,
        "imgsz": imgsz,
        "batch_size": batch_size,
    }
    args_path = output_dir / "args.yaml"
    args_path.write_text(yaml.dump(train_args, default_flow_style=False))

    # Load model and train
    model = YOLO(str(base_model))
    t0 = time.time()

    results = model.train(
        data=str(dataset_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        freeze=freeze,
        lr0=lr0,
        project=str(output_dir),
        name="train",
        exist_ok=True,
        save=True,
        plots=True,
        pose=12.0,  # explicit default — placeholder for future tuning
    )

    elapsed = time.time() - t0

    # Copy best.pt and last.pt to round directory root
    train_dir = output_dir / "train"
    weights_dir = train_dir / "weights" if train_dir.exists() else output_dir

    best_src = weights_dir / "best.pt"
    last_src = weights_dir / "last.pt"
    best_dst = output_dir / "best.pt"
    last_dst = output_dir / "last.pt"

    if best_src.exists():
        shutil.copy2(best_src, best_dst)
    if last_src.exists():
        shutil.copy2(last_src, last_dst)

    # Copy results CSV
    results_csv = train_dir / "results.csv"
    if results_csv.exists():
        shutil.copy2(results_csv, output_dir / "results.csv")

    # Extract metrics
    mAP50 = 0.0
    mAP50_95 = 0.0
    if results is not None:
        box_results = getattr(results, "results_dict", {})
        mAP50 = box_results.get("metrics/mAP50(B)", 0.0)
        mAP50_95 = box_results.get("metrics/mAP50-95(B)", 0.0)

    result = TrainResult(
        model_path=str(best_dst) if best_dst.exists() else str(base_model),
        mAP50=mAP50,
        mAP50_95=mAP50_95,
        training_time_s=elapsed,
        round_num=round_num,
        total_frames=total_frames,
        hyperparameters=train_args,
    )

    logger.info(
        f"Training complete: round {round_num}, "
        f"mAP50={mAP50:.4f}, mAP50-95={mAP50_95:.4f}, "
        f"time={elapsed:.0f}s"
    )
    return result
