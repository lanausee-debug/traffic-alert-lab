"""Train a small CPU PyTorch alert model on independent synthetic timelines."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import random

import torch
from torch import nn

from alert_eval import Sample, evaluate, write_report


def make_timeline(seed: int, length: int = 720):
    """Entire timelines use separate seeds across training/calibration/evaluation."""
    rng = random.Random(seed)
    speeds = [62.0 + rng.gauss(0, 1.2) for _ in range(length)]
    center = 35
    while center < length - 30:
        width = rng.randint(12, 18)
        depth = rng.uniform(40, 48)
        for i in range(max(0, center - width), min(length, center + width + 1)):
            speeds[i] -= depth * max(0, 1 - abs(i - center) / width)
        center += rng.randint(48, 65)
    visible = [rng.random() > 0.15 for _ in speeds]
    features, targets = [], []
    for t in range(12, length - 2):
        # Inputs end at t-1. Targets use t..t+2 only during training/calibration.
        mask = visible[t - 12:t]
        features.append([speeds[i] / 65 if visible[i] else 0.0 for i in range(t - 12, t)]
                        + [float(value) for value in mask])
        targets.append(float(min(speeds[t:t + 3]) <= 30))
    return torch.tensor(features, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32), speeds


def run(output: str | Path, steps: int = 120) -> dict:
    if steps < 1:
        raise ValueError("steps must be positive")
    output = Path(output)
    if output.exists():
        raise FileExistsError("Use a new output directory")
    torch.set_num_threads(1)
    torch.manual_seed(7)
    train_x, train_y, _ = make_timeline(11, 2160)
    cal_x, cal_y, _ = make_timeline(22)
    test_x, _, test_speeds = make_timeline(33)
    model = nn.Sequential(nn.Linear(24, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCEWithLogitsLoss(pos_weight=(1 - train_y).sum() / train_y.sum())
    initial = [parameter.detach().clone() for parameter in model.parameters()]
    history = []
    model.train()
    for _ in range(steps):
        optimizer.zero_grad()
        logits = model(train_x).squeeze(-1)
        loss = criterion(logits, train_y)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite training loss")
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
    model.eval()
    with torch.no_grad():
        cal_scores = model(cal_x).squeeze(-1).sigmoid()
        test_scores = model(test_x).squeeze(-1).sigmoid()
    # Select using calibration labels only. The evaluation timeline is untouched.
    candidates = []
    for threshold in (0.2, 0.35, 0.5, 0.65, 0.8):
        predicted = cal_scores >= threshold
        truth = cal_y.bool()
        tp = int((predicted & truth).sum())
        fp = int((predicted & ~truth).sum())
        fn = int((~predicted & truth).sum())
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        candidates.append({"threshold": threshold, "pointwise_f1": f1})
    selected = max(candidates, key=lambda candidate: (candidate["pointwise_f1"], candidate["threshold"]))
    origin = datetime(2020, 1, 1, tzinfo=timezone.utc)
    rows = [Sample("synthetic-test", origin + timedelta(minutes=5 * t), float(test_scores[t - 12]),
                   int(test_speeds[t] <= 30)) for t in range(12, len(test_speeds) - 2)]
    result = evaluate(rows, threshold=selected["threshold"])
    result["synthetic"] = True
    result["limitations"].append("Synthetic simulation and small MLP are teaching examples, not evidence of traffic performance.")
    write_report(result, output)
    with (output / "predictions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("station_id", "timestamp", "score", "is_event"))
        writer.writerows((row.station_id, row.timestamp.isoformat(), row.score, row.is_event) for row in rows)
    parameters_changed = any(not torch.equal(before, after.detach()) for before, after in zip(initial, model.parameters()))
    if not parameters_changed:
        raise ValueError("Optimizer did not update model parameters")
    record = {"synthetic_only": True, "research_training_performed": False,
        "optimizer_updates": steps, "parameters_changed": parameters_changed,
        "model": "24 -> 32 -> 16 -> 1 MLP", "parameters": sum(p.numel() for p in model.parameters()),
        "timeline_seeds": {"training": 11, "calibration": 22, "evaluation": 33},
        "training_rows": len(train_y), "calibration_rows": len(cal_y), "evaluation_rows": len(rows),
        "input": "past 12 normalized speeds plus 12 observation indicators; 15% missingness",
        "target": "any speed <= 30 mph in t..t+2, for teaching only",
        "loss_history": history, "calibration_candidates": candidates,
        "threshold_selected_without_evaluation_labels": selected["threshold"],
        "evaluation_metrics": result["metrics"], "torch_version": torch.__version__,
        "device": "cpu", "weights_saved": False}
    (output / "training.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New run directory")
    parser.add_argument("--steps", type=int, default=120)
    args = parser.parse_args()
    record = run(args.output, args.steps)
    print(json.dumps({key: value for key, value in record.items() if key != "loss_history"}, indent=2))


if __name__ == "__main__":
    main()
