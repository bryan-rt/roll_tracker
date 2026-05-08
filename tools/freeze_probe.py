"""A/B freeze level probe for Round 2 training decision."""

from ultralytics import YOLO
from pathlib import Path
import json
import time

DATASET = "data/training_data/combined/dataset.yaml"
BASE_MODEL = "models/bjj-pose-r1.pt"  # Round 1 fine-tuned
EPOCHS = 20
DEVICE = "cpu"  # MPS has float64 issue
IMGSZ = 640
BATCH = 8

results = {}

for freeze_level in [10, 6]:
    print(f"\n{'='*60}")
    print(f"PROBE: freeze={freeze_level}, {EPOCHS} epochs")
    print(f"{'='*60}\n")

    output_dir = f"models/training_runs/round2_probe_freeze{freeze_level}"

    model = YOLO(BASE_MODEL)
    t0 = time.time()

    train_results = model.train(
        data=DATASET,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        freeze=freeze_level,
        lr0=0.001 if freeze_level == 10 else 0.0005,
        project=output_dir,
        name="train",
        exist_ok=True,
        save=True,
        plots=True,
        pose=12.0,
    )

    elapsed = time.time() - t0

    # Extract metrics
    rd = getattr(train_results, "results_dict", {})
    metrics = {
        "freeze": freeze_level,
        "epochs": EPOCHS,
        "elapsed_s": round(elapsed),
        "box_mAP50": rd.get("metrics/mAP50(B)", 0.0),
        "box_mAP50_95": rd.get("metrics/mAP50-95(B)", 0.0),
        "pose_mAP50": rd.get("metrics/mAP50(P)", 0.0),
        "pose_mAP50_95": rd.get("metrics/mAP50-95(P)", 0.0),
    }
    results[freeze_level] = metrics

    print(f"\n--- freeze={freeze_level} results ---")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

# Compare
print(f"\n{'='*60}")
print("COMPARISON")
print(f"{'='*60}")
print(f"{'Metric':<20} {'freeze=10':>12} {'freeze=6':>12} {'Winner':>10}")
print("-" * 56)

for metric in ["box_mAP50", "box_mAP50_95", "pose_mAP50", "pose_mAP50_95"]:
    v10 = results[10][metric]
    v6 = results[6][metric]
    winner = "freeze=10" if v10 >= v6 else "freeze=6"
    print(f"{metric:<20} {v10:>12.4f} {v6:>12.4f} {winner:>10}")

# Recommendation
best_freeze = 10 if results[10]["pose_mAP50"] >= results[6]["pose_mAP50"] else 6
print(f"\nRECOMMENDATION: Use freeze={best_freeze} for full training run")
print(f"  Pose mAP50: freeze=10={results[10]['pose_mAP50']:.4f}, freeze=6={results[6]['pose_mAP50']:.4f}")

# Save results
output_path = Path("models/training_runs/round2_probe_results.json")
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(json.dumps(results, indent=2))
print(f"\nResults saved to {output_path}")
