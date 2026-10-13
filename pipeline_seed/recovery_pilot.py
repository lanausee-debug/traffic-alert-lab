"""Small METR-LA recovery-mechanism pilot. No repair-policy novelty claim.

Train on early calendar days, select epoch on validation, probe held-out days.
At a fixed forecast origin, only the repaired station's historical mask changes.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def load_data(path, days, nodes, selection_seed, min_availability):
    with h5py.File(path, "r") as h5:
        raw = h5["df/block0_values"][:].astype(np.float32)
        timestamps = h5["df/axis1"][:].astype("datetime64[ns]")
        ids = h5["df/axis0"][:]
    if raw.shape[0] != len(timestamps):
        raise ValueError("Unexpected HDF5 orientation; inspect file before proceeding")
    if np.any(np.diff(timestamps.astype(np.int64)) <= 0):
        raise ValueError("Timestamps must be strictly increasing")
    day = timestamps.astype("datetime64[D]")
    first_day = day[0]
    boundaries = [0, days * 2 // 3, days * 5 // 6, days]
    cuts = [int(np.searchsorted(day, first_day + np.timedelta64(d, "D"))) for d in boundaries]
    if cuts[-1] > len(raw) or len(np.unique(day)) < days:
        raise ValueError("Not enough calendar days")
    raw, timestamps, day = raw[:cuts[-1]], timestamps[:cuts[-1]], day[:cuts[-1]]
    valid = np.isfinite(raw) & (raw > 0)
    candidates = np.flatnonzero(valid[:cuts[1]].mean(axis=0) >= min_availability)
    if len(candidates) < nodes:
        raise ValueError("Insufficient sensors meeting TRAINING-only availability rule")
    chosen = np.sort(np.random.default_rng(selection_seed).choice(candidates, nodes, replace=False))
    raw, valid = raw[:, chosen], valid[:, chosen]
    counts = valid[:cuts[1]].sum(axis=0)
    mean = np.where(valid[:cuts[1]], raw[:cuts[1]], 0).sum(axis=0) / counts
    variance = np.where(valid[:cuts[1]], (raw[:cuts[1]] - mean) ** 2, 0).sum(axis=0) / counts
    scale = np.maximum(np.sqrt(variance), 1e-3).astype(np.float32)
    mean = mean.astype(np.float32)
    normalized = np.where(valid, (raw - mean) / scale, 0).astype(np.float32)
    corr = np.maximum(np.nan_to_num(np.corrcoef(normalized[:cuts[1]].T)), 0)
    np.fill_diagonal(corr, 0)
    adjacency = np.zeros((nodes, nodes), dtype=np.float32)
    for i in range(nodes):
        neighbors = np.argsort(corr[i])[-min(4, nodes - 1):]
        adjacency[i, neighbors] = corr[i, neighbors]
        adjacency[i, i] = 1
    adjacency /= adjacency.sum(axis=1, keepdims=True)
    minute = (timestamps - day).astype("timedelta64[m]").astype(np.float32)
    weekday = (day.astype(np.int64) + 3) % 7
    context = np.stack([np.sin(2*np.pi*minute/1440), np.cos(2*np.pi*minute/1440),
                        np.sin(2*np.pi*weekday/7), np.cos(2*np.pi*weekday/7)], axis=1).astype(np.float32)
    sensor_ids = [v.decode() if isinstance(v, bytes) else str(v) for v in ids[chosen]]
    return normalized, valid.astype(np.float32), scale, mean, adjacency, timestamps, context, cuts, sensor_ids, chosen


def dataset(z, valid, context, timestamps, start, stop, history, horizon):
    # Entire input and target belong to this partition and have 5-minute spacing.
    origins = [t for t in range(start + history, stop - horizon + 1)
               if np.all(np.diff(timestamps[t-history:t+horizon]) == np.timedelta64(5, "m"))]
    if not origins:
        raise ValueError("No valid contiguous windows")
    return TensorDataset(*[torch.from_numpy(v) for v in (
        np.stack([z[t-history:t] for t in origins]),
        np.stack([valid[t-history:t] for t in origins]),
        np.stack([z[t:t+horizon] for t in origins]),
        np.stack([valid[t:t+horizon] for t in origins]), context[origins])])


def artificial_mask(base, generator, history):
    b, _, n = base.shape
    disrupted = torch.rand((b, n), generator=generator) < .30
    count = torch.randint(0, history + 1, (b, n), generator=generator)
    positions = torch.arange(history).view(1, history, 1)
    available = (~disrupted[:, None, :]) | (positions >= history - count[:, None, :])
    point = torch.rand(base.shape, generator=generator) >= .03
    return base * available * point


def latest(z, mask):
    positions = torch.arange(z.shape[1]).view(1, -1, 1)
    index = torch.where(mask > 0, positions, -1).amax(dim=1)
    values = z.gather(1, index.clamp_min(0).unsqueeze(1)).squeeze(1)
    return torch.where(index >= 0, values, torch.zeros_like(values))


class RecoveryForecaster(nn.Module):
    def __init__(self, history, horizon, adjacency, persistence_residual=True):
        super().__init__()
        self.register_buffer("adjacency", torch.tensor(adjacency))
        self.persistence_residual = persistence_residual
        self.mlp = nn.Sequential(nn.Linear(history * 4 + 4, 64), nn.ReLU(), nn.Linear(64, horizon))

    def forward(self, z, mask, context):
        values = z * mask
        coverage = torch.einsum("ij,btj->bti", self.adjacency, mask)
        neighbors = torch.einsum("ij,btj->bti", self.adjacency, values) / coverage.clamp_min(1e-6)
        features = torch.stack([values, mask, neighbors, coverage], dim=-1).permute(0, 2, 1, 3).flatten(2)
        ctx = context[:, None, :].expand(-1, z.shape[2], -1)
        correction = self.mlp(torch.cat([features, ctx], dim=-1)).permute(0, 2, 1)
        return latest(z, mask)[:, None, :] + correction if self.persistence_residual else correction


def validation(model, data, fixed_mask, scale, batch_size=128):
    total, baseline, count = 0., 0., 0.
    model.eval()
    loader = DataLoader(TensorDataset(*data.tensors, fixed_mask), batch_size=batch_size)
    with torch.no_grad():
        for x, natural, y, target_valid, context, mask in loader:
            error = ((model(x, mask, context) - y).abs() * scale) * target_valid
            base = ((latest(x, mask)[:, None, :] - y).abs() * scale) * target_valid
            total += error.sum().item()
            baseline += base.sum().item()
            count += target_valid.sum().item()
    return {"model_mae": total/count, "last_observed_mae": baseline/count, "valid_targets": int(count)}


def train(seed, data, val, scale, adjacency, args, output):
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    model = RecoveryForecaster(args.history, args.horizon, adjacency, not args.no_persistence_residual)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    generator = torch.Generator().manual_seed(seed + 100)
    shuffle = torch.Generator().manual_seed(seed + 200)
    fixed_mask = artificial_mask(val.tensors[1], torch.Generator().manual_seed(991), args.history)
    loader = DataLoader(data, batch_size=128, shuffle=True, generator=shuffle)
    best, best_state, best_epoch = float("inf"), None, None
    logs = []
    for epoch in range(1, args.epochs+1):
        model.train()
        for x, natural, y, target_valid, context in loader:
            mask = artificial_mask(natural, generator, args.history)
            optimizer.zero_grad(set_to_none=True)
            error = (model(x, mask, context) - y).abs() * target_valid
            loss = error.sum() / target_valid.sum().clamp_min(1)
            loss.backward()
            optimizer.step()
        metrics = validation(model, val, fixed_mask, scale)
        logs.append({"epoch": epoch, **metrics})
        if metrics["model_mae"] < best:
            best = metrics["model_mae"]
            best_epoch = epoch
            best_state = {k:v.detach().clone() for k,v in model.state_dict().items()}
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(f"seed={seed} epoch={epoch} valMAE={metrics['model_mae']:.4f} persistence={metrics['last_observed_mae']:.4f}", flush=True)
    model.load_state_dict(best_state)
    model.eval()
    torch.save({"state_dict": best_state, "seed": seed, "best_epoch": best_epoch}, output/f"model_seed{seed}.pt")
    return model, {"seed": seed, "best_epoch": best_epoch, "selected_validation_mae": best, "training": logs}


def probe_events(z, valid, timestamps, context, cuts, history, horizon, nodes):
    days = timestamps.astype("datetime64[D]")
    test_days = np.unique(days[cuts[2]:cuts[3]])
    selected_faults = np.sort(np.random.default_rng(2026).choice(nodes, min(6, nodes), replace=False))
    events = []
    for date in test_days:
        for hour in (6, 12, 18):
            desired = date.astype("datetime64[ns]") + np.timedelta64(hour, "h")
            t = int(np.searchsorted(timestamps, desired))
            if t >= len(z) or timestamps[t] != desired or t-history < cuts[2] or t+horizon > cuts[3]:
                continue
            if not np.all(np.diff(timestamps[t-history:t+horizon]) == np.timedelta64(5,"m")):
                continue
            for node in selected_faults:
                if not valid[t:t+horizon, node].any():
                    continue
                events.append({"origin": t, "node": int(node), "day": str(date), "hour": hour})
    if not events:
        raise ValueError("No test probe events")
    return events


def probe(model, events, z, valid, context, scale, args):
    x = torch.from_numpy(np.stack([z[e['origin']-args.history:e['origin']] for e in events]))
    natural = torch.from_numpy(np.stack([valid[e['origin']-args.history:e['origin']] for e in events]))
    y = torch.from_numpy(np.stack([z[e['origin']:e['origin']+args.horizon] for e in events]))
    target_valid = torch.from_numpy(np.stack([valid[e['origin']:e['origin']+args.horizon] for e in events]))
    ctx = torch.from_numpy(np.stack([context[e['origin']] for e in events]))
    sensors = torch.tensor([e['node'] for e in events])
    rows = torch.arange(len(events))
    # Same forecast origin, target, neighbors and weights for every recovery age.
    with torch.no_grad():
        clean_prediction = model(x, natural, ctx)
        broken_mask = natural.clone()
        broken_mask[rows, :, sensors] = 0
        broken_prediction = model(x, broken_mask, ctx)
        hidden_values_changed = torch.where(broken_mask > 0, x, x + 1234)
        if not torch.equal(broken_prediction, model(hidden_values_changed, broken_mask, ctx)):
            raise AssertionError("Predictions must be invariant to values behind a missing-data mask")

        def loss(prediction):
            error = (prediction-y).abs() * scale
            per_node = (error * target_valid).sum(dim=1) / target_valid.sum(dim=1).clamp_min(1)
            return per_node[rows, sensors].numpy()

        broken, clean = loss(broken_prediction), loss(clean_prediction)
        recovered, global_errors = [], []
        for age in range(args.history + 1):
            mask = broken_mask.clone()
            if age:
                mask[rows, -age:, sensors] = natural[rows, -age:, sensors]
            prediction = model(x, mask, ctx)
            recovered.append(loss(prediction))
            global_errors.append(((prediction-y).abs()*scale*target_valid).sum().item()/target_valid.sum().item())
            if age == 0 and not torch.equal(prediction, broken_prediction):
                raise AssertionError("Age zero must equal ongoing outage")
            if age == args.history and not torch.equal(prediction, clean_prediction):
                raise AssertionError("A full restored window must equal the ideal-history input")
    return np.stack(recovered, axis=1), broken, clean, global_errors


def paired_day_interval(gains, events):
    days = sorted({e['day'] for e in events})
    daily = np.stack([gains[[i for i,e in enumerate(events) if e['day']==d]].mean(axis=0) for d in days])
    rng = np.random.default_rng(812)
    samples = daily[rng.integers(0, len(days), size=(2000, len(days)))].mean(axis=1)
    return daily.mean(axis=0), np.quantile(samples, [.025, .975], axis=0), days


def scheduling_sanity():
    # Constructed values, NOT inferred from METR-LA and NOT empirical improvement.
    def evaluate(order, changing):
        t, completion = 0, {}
        duration = {"A": 2, "B": 1}
        for station in order:
            t += duration[station]
            completion[station] = t
        gain = 0
        for hour in range(4):
            for station in ("A", "B"):
                if hour >= completion[station]:
                    weight = 10 if station == "A" else (20 if changing and hour >= 1 else 1)
                    gain += weight
        return gain
    return {"source": "constructed two-station illustration; no learned policy",
            "constant_values": {"A_then_B": evaluate(("A","B"), False), "B_then_A": evaluate(("B","A"), False)},
            "changing_B_value": {"A_then_B": evaluate(("A","B"), True), "B_then_A": evaluate(("B","A"), True)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT/"data/raw/metr-la.h5")
    parser.add_argument("--output", type=Path, default=ROOT/"outputs/recovery-pilot")
    parser.add_argument("--days", type=int, default=42)
    parser.add_argument("--nodes", type=int, default=24)
    parser.add_argument("--history", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--min-train-availability", type=float, default=.90)
    parser.add_argument("--no-persistence-residual", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42,43,44])
    args = parser.parse_args()
    if args.days < 12 or args.nodes < 3 or min(args.history, args.horizon, args.epochs) < 1:
        parser.error("Need >=12 days, >=3 sensors, positive history/horizon/epochs")
    if not 0 < args.min_train_availability <= 1:
        parser.error("Training availability must be in (0, 1]")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    started = time.perf_counter()
    z, valid, scale, mean, adjacency, timestamps, context, cuts, ids, columns = load_data(args.data, args.days, args.nodes, 1701, args.min_train_availability)
    scale_tensor = torch.tensor(scale).view(1,1,-1)
    train_data = dataset(z, valid, context, timestamps, cuts[0], cuts[1], args.history, args.horizon)
    val_data = dataset(z, valid, context, timestamps, cuts[1], cuts[2], args.history, args.horizon)
    events = probe_events(z,valid,timestamps,context,cuts,args.history,args.horizon,args.nodes)
    config = {
        "data": str(args.data.resolve()), "days": args.days, "sensors": args.nodes,
        "history_steps": args.history, "horizon_steps": args.horizon, "sampling_minutes": 5,
        "epochs": args.epochs, "training_seeds": args.seeds, "selection_seed": 1701, "fault_selection_seed": 2026,
        "calendar_partition_days": [args.days*2//3, args.days*5//6-args.days*2//3, args.days-args.days*5//6],
        "time_ranges_half_open": {name:[str(timestamps[a]),str(timestamps[b-1]+np.timedelta64(5,'m'))]
                                  for name,a,b in zip(('train','validation','test'),cuts[:-1],cuts[1:])},
        "selected_sensor_ids": ids, "raw_column_indices": columns.tolist(),
        "graph": "4 nearest positive-correlation neighbors plus self; fitted on TRAIN ONLY; not a geographic road graph",
        "normalization_mean": mean.tolist(), "normalization_std": scale.tolist(),
        "unit": "original METR-LA speed values; no conversion", "zero_handling": "zero/nonfinite values are missing; targets excluded",
        "selection_rule": f"random {args.nodes} sensors from >={args.min_train_availability:.0%} training availability; no outcome-based selection",
        "availability_rule_note": "Initial 95% rule yielded zero eligible sensors in training data; lowered to 90% BEFORE any model fitting or test analysis.",
        "model": "fixed masked graph aggregation + shared temporal MLP + time context; NOT Graph WaveNet",
        "persistence_residual": not args.no_persistence_residual,
        "probe": "same forecast origin and future targets across all recovery ages; one artificial failed station per event",
        "train_windows": len(train_data), "validation_windows":len(val_data), "probe_events":len(events),
        "notes": ["Real speed data + artificial persistent outages; no field maintenance records.",
                  "Ideal history is a diagnostic only; never a deployable repair policy.",
                  "No test labels used to train/select a repair policy; this run implements NO learned repair policy.",
                  "No pre-specified effect-size success threshold; exploratory pilot, not confirmatory evidence."]}
    # Configuration is written before fitting/testing, and not changed based on test gains.
    write_json(args.output/"config.json",config)
    write_json(args.output/"events.json",events)
    all_recovered, all_broken, all_clean, training, all_global = [],[],[],[],[]
    print(f"METR-LA rows={len(z)} sensors={args.nodes} partitions={config['calendar_partition_days']} events={len(events)}",flush=True)
    for seed in args.seeds:
        model,log = train(seed,train_data,val_data,scale_tensor,adjacency,args,args.output)
        recovered,broken,clean,global_errors = probe(model,events,z,valid,context,scale_tensor,args)
        all_recovered.append(recovered); all_broken.append(broken); all_clean.append(clean); training.append(log); all_global.append(global_errors)
    recovered = np.stack(all_recovered).mean(axis=0)
    broken = np.stack(all_broken).mean(axis=0)
    clean = np.stack(all_clean).mean(axis=0)
    gains = broken[:,None]-recovered
    mean_gain,interval,test_days = paired_day_interval(gains,events)
    groups = [[i for i,e in enumerate(events) if e['day']==d] for d in test_days]
    def day_macro(values):
        return np.stack([values[group].mean(axis=0) for group in groups]).mean(axis=0)
    recovered_mean = day_macro(recovered)
    broken_mean, clean_mean = float(day_macro(broken)), float(day_macro(clean))
    remaining_gain, remaining_interval, _ = paired_day_interval(gains[:,-1:]-gains[:,min(2,args.history):min(2,args.history)+1], events)
    curves = []
    for age in range(args.history+1):
        curves.append({"recovered_minutes":age*5,"usable_new_slots":age,"failed_station_mae":float(recovered_mean[age]),
                       "ongoing_outage_mae":broken_mean,"ideal_history_mae":clean_mean,
                       "gain_vs_outage":float(mean_gain[age]),"gain_ci_low":float(interval[0,age]),"gain_ci_high":float(interval[1,age]),
                       "all_station_mae":float(np.mean(all_global,axis=0)[age])})
    per_event=[]
    for i,e in enumerate(events):
        per_event.append({**e,"sensor_id":ids[e['node']],"outage_mae":float(broken[i]),"ideal_mae":float(clean[i]),
                          "gain_5min":float(gains[i,1]),"gain_full_window":float(gains[i,-1])})
    worse_steps = int(np.sum(np.diff(recovered_mean)>1e-6))
    result={"config":config,"curves":curves,"training_runs":training,"test_days":test_days,
            "full_window_gain":float(mean_gain[-1]),"full_window_gain_ci":interval[:,-1].tolist(),
            "full_window_relative_mae_reduction":float(mean_gain[-1]/broken_mean),
            "gain_after_first_reading":float(mean_gain[1]),"aggregate_error_increasing_steps":worse_steps,
            "full_gain_fraction_available_at_10min":float(mean_gain[min(2,args.history)]/mean_gain[-1]) if mean_gain[-1] else None,
            "remaining_gain_after_10min":float(remaining_gain[0]),"remaining_gain_after_10min_ci":remaining_interval[:,0].tolist(),
            "aggregation": "primary failed-station errors and gains: equal weight per measurable test day, model seeds averaged first; all_station_mae is pooled",
            "per_seed_full_gain": [{"seed":seed,"gain":float(day_macro(b-c)),"outage_mae":float(day_macro(b)),"ideal_mae":float(day_macro(c))}
                                   for seed,b,c in zip(args.seeds,all_broken,all_clean)],
            "events_with_positive_full_gain":int(np.sum(gains[:,-1]>0)),"per_event":per_event,
            "scheduling_sanity":scheduling_sanity(),"elapsed_seconds":time.perf_counter()-started,
            "conclusion_limits":["This measures input availability effects for ONE small predictor on ONE small data slice.",
                                  "It does not show better physical repair scheduling, actual congestion control, or original algorithm novelty.",
                                  "Confidence intervals resample held-out days after averaging model seeds; small pilot intervals are exploratory.",
                                  "The two-station scheduling example is constructed and separate from the real-data experiment."]}
    write_json(args.output/"results.json",result)
    with (args.output/"recovery_curve.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(curves[0])); writer.writeheader(); writer.writerows(curves)
    with (args.output/"episode_results.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(per_event[0])); writer.writeheader(); writer.writerows(per_event)
    x=np.arange(args.history+1)*5
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout="constrained")
    axes[0].plot(x,recovered_mean,"o-",label="New observations only")
    axes[0].axhline(broken_mean,color="#bf4d45",linestyle="--",label="Ongoing outage")
    axes[0].axhline(clean_mean,color="#40936c",linestyle=":",label="Ideal full history (diagnostic)")
    axes[0].set(xlabel="Time already restored at fixed forecast origin (min)",ylabel="Failed-station MAE (original speed units)",title="Real METR-LA + artificial outages")
    axes[0].legend(fontsize=8)
    axes[1].plot(x,mean_gain,"o-",color="#287bb5")
    axes[1].fill_between(x,interval[0],interval[1],alpha=.18,color="#287bb5")
    axes[1].axhline(0,color="gray",linewidth=.7)
    axes[1].set(xlabel="Time already restored (min)",ylabel="Paired MAE reduction vs ongoing outage",title=f"Exploratory 95% interval over {len(test_days)} held-out days")
    for ax in axes: ax.grid(alpha=.15)
    fig.savefig(args.output/"recovery_curve.png",dpi=170)
    fig.savefig(args.output/"recovery_curve.svg")
    plt.close(fig)
    print(json.dumps({"outage_mae":broken_mean,"ideal_mae":clean_mean,"full_gain":float(mean_gain[-1]),
                      "full_gain_ci":interval[:,-1].tolist(),"positive_events":result['events_with_positive_full_gain'],
                      "events":len(events),"nonmonotone_steps":worse_steps,"gain_fraction_at_10min":result['full_gain_fraction_available_at_10min'],
                      "remaining_gain_ci":result['remaining_gain_after_10min_ci'],"elapsed":result['elapsed_seconds']},indent=2),flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
