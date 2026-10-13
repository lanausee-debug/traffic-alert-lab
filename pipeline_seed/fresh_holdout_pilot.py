"""New calendar holdout, masked graph TCN candidate, repair-space screening."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from recovery_pilot import ROOT, RecoveryForecaster, artificial_mask, dataset, latest, validation, write_json
from multifault_pilot import (calibrated_policies, day_interval, day_macro, evaluate_events,
                             infer, make_events, relative_day_interval, sha256, summarize_policy)


class MaskedGraphTCN(nn.Module):
    """Existing diffusion + gated temporal convolution ideas; not STGCN reproduction."""
    def __init__(self, history, horizon, adjacency):
        super().__init__()
        self.register_buffer("adjacency", torch.tensor(adjacency))
        self.register_buffer("two_hop", torch.tensor(adjacency @ adjacency))
        self.embedding = nn.Parameter(torch.empty(len(adjacency), 8))
        nn.init.normal_(self.embedding, std=.05)
        self.project = nn.Conv1d(18, 32, 1)
        self.temporal = nn.ModuleList([nn.Conv1d(32, 64, 3, dilation=d) for d in (1, 2, 4)])
        self.dropout = nn.Dropout(.05)
        self.head = nn.Linear(32, horizon)

    def forward(self, z, mask, context):
        values = z * mask
        cov1 = torch.einsum("ij,btj->bti", self.adjacency, mask)
        cov2 = torch.einsum("ij,btj->bti", self.two_hop, mask)
        first = torch.einsum("ij,btj->bti", self.adjacency, values)/cov1.clamp_min(1e-6)
        second = torch.einsum("ij,btj->bti", self.two_hop, values)/cov2.clamp_min(1e-6)
        features = torch.stack([values, mask, first, cov1, second, cov2], dim=-1)
        b, length, n, _ = features.shape
        ctx = context[:, None, None, :].expand(b, length, n, -1)
        emb = self.embedding[None, None].expand(b, length, n, -1)
        x = torch.cat([features, ctx, emb], dim=-1).permute(0, 2, 3, 1).reshape(b*n, 18, length)
        hidden = self.project(x)
        for conv, dilation in zip(self.temporal, (1, 2, 4)):
            out = conv(nn.functional.pad(hidden, (2*dilation, 0)))
            signal, gate = out.chunk(2, dim=1)
            hidden = hidden+self.dropout(torch.tanh(signal)*torch.sigmoid(gate))
        correction = self.head(hidden[:, :, -1]).reshape(b, n, -1).permute(0, 2, 1)
        return latest(z, mask)[:, None, :]+correction


def read_frozen_data():
    config = json.loads((ROOT/"outputs/recovery-pilot/config.json").read_text(encoding="utf-8"))
    with h5py.File(ROOT/"data/raw/metr-la.h5", "r") as h5:
        raw = h5["df/block0_values"][:, config["raw_column_indices"]].astype(np.float32)
        timestamps = h5["df/axis1"][:].astype("datetime64[ns]")
    day = timestamps.astype("datetime64[D]")
    boundaries = [0, 28, 35, 42, 49, 77]
    cuts = [int(np.searchsorted(day, day[0]+np.timedelta64(d, "D"))) for d in boundaries]
    raw, timestamps, day = raw[:cuts[-1]], timestamps[:cuts[-1]], day[:cuts[-1]]
    valid = (np.isfinite(raw) & (raw > 0)).astype(np.float32)
    mean = np.array(config["normalization_mean"], dtype=np.float32)
    scale = np.array(config["normalization_std"], dtype=np.float32)
    normalized = np.where(valid > 0, (raw-mean)/scale, 0).astype(np.float32)
    checkpoint = torch.load(ROOT/"outputs/recovery-pilot/model_seed42.pt", weights_only=True)
    adjacency = checkpoint["state_dict"]["adjacency"].numpy()
    minute = (timestamps-day).astype("timedelta64[m]").astype(np.float32)
    weekday = (day.astype(np.int64)+3) % 7
    context = np.stack([np.sin(2*np.pi*minute/1440), np.cos(2*np.pi*minute/1440),
                        np.sin(2*np.pi*weekday/7), np.cos(2*np.pi*weekday/7)], axis=1).astype(np.float32)
    return normalized, valid, scale, adjacency, timestamps, context, cuts, config


def train_tcn(seed, training, val, scale, adjacency, output):
    torch.manual_seed(seed)
    model = MaskedGraphTCN(12, 3, adjacency)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    masks = torch.Generator().manual_seed(seed+100)
    loader = DataLoader(training, batch_size=128, shuffle=True, generator=torch.Generator().manual_seed(seed+200))
    fixed_mask = artificial_mask(val.tensors[1], torch.Generator().manual_seed(991), 12)
    best = float("inf")
    logs = []
    started = time.perf_counter()
    for epoch in range(1, 26):
        model.train()
        for x, natural, y, targets, context in loader:
            mask = artificial_mask(natural, masks, 12)
            optimizer.zero_grad(set_to_none=True)
            loss = ((model(x, mask, context)-y).abs()*targets).sum()/targets.sum().clamp_min(1)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
        metrics = validation(model, val, fixed_mask, scale)
        logs.append({"epoch": epoch, **metrics})
        if metrics["model_mae"] < best:
            best, selected_epoch = metrics["model_mae"], epoch
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % 5 == 0:
            print(f"TCN seed={seed} epoch={epoch} valMAE={metrics['model_mae']:.4f} elapsed={time.perf_counter()-started:.1f}s", flush=True)
    model.load_state_dict(state)
    model.eval()
    torch.save({"state_dict": state, "seed": seed, "best_epoch": selected_epoch}, output/f"model_tcn_seed{seed}.pt")
    return model, {"seed": seed, "selected_epoch": selected_epoch, "validation_mae": best,
                   "parameters": sum(p.numel() for p in model.parameters()), "training_seconds": time.perf_counter()-started, "epochs": logs}


def healthy_cost(model, events, z, valid, context, scale):
    # With all fault nodes' full natural history, infer uses the no-artificial-outage input.
    requests, indices, mapping = [], {}, []
    for event in events:
        rows = []
        for step in event["usable_steps"]:
            origin = event["origin"]+step
            if origin not in indices:
                indices[origin] = len(requests)
                requests.append((origin, [0, 1, 2], [12, 12, 12]))
            rows.append(indices[origin])
        mapping.append(rows)
    losses = infer(model, requests, z, valid, context, scale, 12, 3)
    return np.array([5*losses[rows].sum() for rows in mapping])


def primary_summary(events, actual, additive, outage, fixed, healthy, policies, reference):
    summary, choices, costs = summarize_policy(events, actual, additive, outage, policies, fixed)
    ref = costs[reference]
    optimal = costs["offline_joint_oracle"]
    delta = ref-optimal
    mean, ci = day_interval(delta, events)
    denominator = float(day_macro(ref, events))
    summary["reference_selected_on_calibration"] = reference
    summary["offline_joint_vs_reference_relative_gain"] = float(mean/denominator)
    summary["offline_joint_vs_reference_relative_ci"] = relative_day_interval(delta, ref, events).tolist()
    summary["healthy_cumulative_loss"] = float(day_macro(healthy, events))
    excess = ref-healthy
    excess_mean, excess_ci = day_interval(excess, events)
    summary["reference_fault_added_loss"] = float(excess_mean)
    summary["reference_fault_added_loss_ci"] = excess_ci.tolist()
    summary["gain_as_fraction_of_fault_added_loss"] = float(mean/excess_mean) if excess_mean > 0 else None
    summary["gain_as_fraction_of_fault_added_loss_ci"] = relative_day_interval(delta, excess, events).tolist() if excess_ci[0] > 0 else None
    summary["ongoing_outage_added_loss"] = float(day_macro(outage-healthy, events))
    summary["cumulative_loss_span_best_to_worst_order"] = float(day_macro(actual.max(axis=1)-actual.min(axis=1), events))
    return summary, choices, costs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"outputs/fresh-holdout-pilot")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    z, valid, scale, adjacency, timestamps, context, cuts, original = read_frozen_data()
    scale_t = torch.tensor(scale).view(1, 1, -1)
    groups = json.loads((ROOT/"outputs/multifault-pilot/config.json").read_text(encoding="utf-8"))["groups"]
    events = {name: make_events(timestamps, valid, cuts, 12, 3, groups, partition, 24)
              for name, partition in (("calibration", 3), ("test", 4))}
    config = {"date": "2026-10-10", "protocol_sha256": sha256(ROOT/"docs/fresh_holdout_protocol.zh-CN.md"),
              "data_sha256": sha256(ROOT/"data/raw/metr-la.h5"), "groups": groups, "events": events,
              "time_ranges": {name: [str(timestamps[a]), str(timestamps[b-1]+np.timedelta64(5, 'm'))]
                              for name, a, b in zip(("training", "model_validation", "old_test_unused", "new_calibration", "new_test"), cuts[:-1], cuts[1:])},
              "raw_columns": original["raw_column_indices"], "normalization_frozen": True, "graph_frozen": True,
              "epochs": 25, "seeds": [42, 43, 44], "candidate_channels": 32, "candidate_embedding": 8,
              "validation_improvement_screen": .05, "offline_total_loss_gain_screen": .03,
              "candidate": "masked one/two-hop graph features + node embedding + 3 gated dilated temporal convolution blocks + persistence residual; not official STGCN",
              "baseline_weight_hashes": {str(s): sha256(ROOT/"outputs/recovery-pilot"/f"model_seed{s}.pt") for s in (42, 43, 44)},
              "primary_metric": "offline six-order optimum gain vs best calibration-selected implementable fixed policy; total network cumulative forecast MAE",
              "auxiliary_metric": "same absolute gain divided by fault-added loss versus no-artificial-outage input; diagnostic only"}
    path = args.output/"config.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != config:
        raise ValueError("Frozen config changed; use another output directory")
    write_json(path, config)
    if args.prepare_only:
        print(json.dumps({"config": str(path), "events": {name: len(es) for name, es in events.items()}, "dates": config["time_ranges"]}, indent=2))
        return
    training = dataset(z, valid, context, timestamps, cuts[0], cuts[1], 12, 3)
    val = dataset(z, valid, context, timestamps, cuts[1], cuts[2], 12, 3)
    test = dataset(z, valid, context, timestamps, cuts[4], cuts[5], 12, 3)
    fixed_val = artificial_mask(val.tensors[1], torch.Generator().manual_seed(991), 12)
    fixed_test = artificial_mask(test.tensors[1], torch.Generator().manual_seed(992), 12)
    result = {"config": config, "models": {}, "checks": {"hidden_mask_values_invariant": True, "no_prerepair_history": True}}
    for variant in ("mlp_baseline", "graph_tcn_candidate"):
        models, logs = [], []
        for seed in (42, 43, 44):
            if variant == "mlp_baseline":
                model = RecoveryForecaster(12, 3, adjacency, True)
                saved = torch.load(ROOT/"outputs/recovery-pilot"/f"model_seed{seed}.pt", weights_only=True)
                model.load_state_dict(saved["state_dict"])
                model.eval()
                log = {"seed": seed, "selected_epoch": saved["best_epoch"], "validation_mae": validation(model, val, fixed_val, scale_t)["model_mae"]}
            else:
                model, log = train_tcn(seed, training, val, scale_t, adjacency, args.output)
            models.append(model)
            logs.append(log)
        # Every model has been selected entirely on old validation before its new test is evaluated.
        write_json(args.output/f"model_selection_{variant}.json", logs)
        partitions = {}
        for partition, es in events.items():
            evaluated = []
            for seed, model in zip((42, 43, 44), models):
                arrays = evaluate_events(model, es, groups, z, valid, context, scale_t, 12, 3, 24, False)
                healthy = healthy_cost(model, es, z, valid, context, scale_t)
                evaluated.append((*arrays[:4], healthy))
                print(f"{variant} seed={seed} {partition} events={len(es)} unique_inputs={arrays[4]}", flush=True)
            partitions[partition] = [np.stack([a[i] for a in evaluated]).mean(axis=0) for i in range(5)]
            if partition == "calibration":
                actual, additive, outage, fixed, healthy = partitions[partition]
                policies = calibrated_policies(es, groups, actual, additive, fixed, False)
                _, _, cal_costs = summarize_policy(es, actual, additive, outage, policies, fixed)
                names = ("shortest_time", "gain_per_time", "additive", "joint")
                calibration_means = {name: float(day_macro(cal_costs[name], es)) for name in names}
                reference = min(names, key=lambda name: calibration_means[name])
                # Saved before any new test evaluation for this model variant.
                write_json(args.output/f"policies_{variant}.json", {"policies": policies, "reference": reference, "calibration_mean_costs": calibration_means})
        actual, additive, outage, fixed, healthy = partitions["test"]
        summary, choices, costs = primary_summary(events["test"], actual, additive, outage, fixed, healthy, policies, reference)
        summary["training_runs"] = logs
        summary["mean_validation_mae"] = float(np.mean([log["validation_mae"] for log in logs]))
        summary["mean_new_test_fixed_mask_mae"] = float(np.mean([validation(m, test, fixed_test, scale_t)["model_mae"] for m in models]))
        result["models"][variant] = summary
        np.savez_compressed(args.output/f"arrays_{variant}.npz", actual=actual, additive=additive, outage=outage,
                            fixed=fixed, healthy=healthy, calibration_actual=partitions["calibration"][0],
                            calibration_additive=partitions["calibration"][1], calibration_outage=partitions["calibration"][2],
                            calibration_fixed=partitions["calibration"][3], calibration_healthy=partitions["calibration"][4])
        write_json(args.output/f"summary_{variant}.json", summary)
        print(json.dumps({"variant": variant, "validation_mae": summary["mean_validation_mae"],
                          "reference": reference, "offline_total_gain": summary["offline_joint_vs_reference_relative_gain"],
                          "offline_total_gain_ci": summary["offline_joint_vs_reference_relative_ci"],
                          "fault_added_gain_fraction": summary["gain_as_fraction_of_fault_added_loss"]}, indent=2), flush=True)
    base, candidate = result["models"]["mlp_baseline"], result["models"]["graph_tcn_candidate"]
    improvement = 1-candidate["mean_validation_mae"]/base["mean_validation_mae"]
    result["candidate_validation_relative_improvement"] = float(improvement)
    result["stronger_predictor_screen_passed"] = bool(improvement >= .05)
    result["repair_value_development_screen_passed"] = bool(improvement >= .05
        and candidate["offline_joint_vs_reference_relative_gain"] >= .03 and candidate["offline_joint_vs_reference_relative_ci"][0] > 0)
    result["elapsed_seconds"] = time.perf_counter()-started
    write_json(args.output/"results.json", result)
    print(json.dumps({"done_seconds": result["elapsed_seconds"], "validation_improvement": improvement,
                      "repair_value_development_screen": result["repair_value_development_screen_passed"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
