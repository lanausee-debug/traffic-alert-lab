"""Frozen-predictor, multi-fault mechanism and sequential repair pilot.

Calibration uses validation dates only. Test oracle choices are diagnostics.
See docs/multifault_protocol.zh-CN.md, frozen before this experiment.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np
import torch

from recovery_pilot import ROOT, RecoveryForecaster, load_data, write_json

ORDERS = list(itertools.permutations(range(3)))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def choose_groups(adjacency):
    support = (adjacency > 0).astype(np.int64)
    overlap = support.T @ support
    candidates = list(itertools.combinations(range(len(adjacency)), 3))
    scores = {g: sum(int(overlap[i, j]) for i, j in itertools.combinations(g, 2)) for g in candidates}
    rng = np.random.default_rng(1010)
    selected = []
    selected += [(g, "high_overlap") for g in sorted(candidates, key=lambda g: (-scores[g], g))[:4]]
    zero = [g for g in candidates if scores[g] == 0]
    if len(zero) < 4:
        raise ValueError("Need four disjoint-output triples for structural checks")
    selected += [(zero[int(i)], "disjoint_outputs") for i in rng.choice(len(zero), 4, replace=False)]
    remaining = [g for g in candidates if g not in {s[0] for s in selected}]
    selected += [(remaining[int(i)], "random") for i in rng.choice(len(remaining), 4, replace=False)]
    return [{"id": k, "nodes": list(g), "category": c, "shared_output_pair_count": scores[g],
             "durations_steps": rng.permutation([2, 4, 6]).tolist()} for k, (g, c) in enumerate(selected)]


def make_events(timestamps, valid, cuts, history, horizon, groups, partition, steps):
    start, stop = cuts[partition], cuts[partition + 1]
    events = []
    for date in np.unique(timestamps[start:stop].astype("datetime64[D]")):
        for hour in (6, 12, 18):
            wanted = date.astype("datetime64[ns]") + np.timedelta64(hour, "h")
            origin = int(np.searchsorted(timestamps, wanted))
            if origin >= stop or timestamps[origin] != wanted or origin-history < start or origin+steps+horizon-1 > stop:
                continue
            if not np.all(np.diff(timestamps[origin-history:origin+steps+horizon-1]) == np.timedelta64(5, "m")):
                continue
            target_counts = np.array([valid[origin+s:origin+s+horizon].sum() for s in range(steps)])
            usable = np.flatnonzero(target_counts > 0).tolist()
            if not usable:
                continue
            for group in groups:
                events.append({"origin": origin, "day": str(date), "hour": hour, "group": group["id"],
                               "category": group["category"], "usable_steps": usable,
                               "mechanism_usable": bool(target_counts[0] > 0)})
    if not events:
        raise ValueError("No measurable events")
    return events


def day_macro(values, events):
    days = sorted({e["day"] for e in events})
    return np.stack([values[[i for i, e in enumerate(events) if e["day"] == d]].mean(axis=0) for d in days]).mean(axis=0)


def day_interval(values, events):
    days = sorted({e["day"] for e in events})
    daily = np.stack([values[[i for i, e in enumerate(events) if e["day"] == d]].mean(axis=0) for d in days])
    rng = np.random.default_rng(812)
    sampled = daily[rng.integers(0, len(days), size=(2000, len(days)))].mean(axis=1)
    return daily.mean(axis=0), np.quantile(sampled, [.025, .975], axis=0)


def relative_day_interval(difference, baseline, events):
    """Resample numerator and denominator together for a ratio of means."""
    days = sorted({e["day"] for e in events})
    daily_difference = np.array([difference[[i for i, e in enumerate(events) if e["day"] == d]].mean() for d in days])
    daily_baseline = np.array([baseline[[i for i, e in enumerate(events) if e["day"] == d]].mean() for d in days])
    indexes = np.random.default_rng(812).integers(0, len(days), size=(2000, len(days)))
    ratios = daily_difference[indexes].mean(axis=1)/daily_baseline[indexes].mean(axis=1)
    return np.quantile(ratios, [.025, .975])


def infer(model, requests, z, valid, context, scale, history, horizon, chunk=256):
    """Request: (origin, three fault nodes, restored suffix lengths).

    Entire historical value and future target tensors are used by the evaluator.
    Prediction only sees masked history; target access never enters a policy.
    """
    losses = []
    with torch.no_grad():
        for begin in range(0, len(requests), chunk):
            batch = requests[begin:begin+chunk]
            x = torch.from_numpy(np.stack([z[t-history:t] for t, _, _ in batch]))
            natural = torch.from_numpy(np.stack([valid[t-history:t] for t, _, _ in batch]))
            mask = natural.clone()
            for row, (_, nodes, ages) in enumerate(batch):
                for node, age in zip(nodes, ages):
                    mask[row, :, node] = 0
                    if age:
                        mask[row, -min(age, history):, node] = natural[row, -min(age, history):, node]
            ctx = torch.from_numpy(np.stack([context[t] for t, _, _ in batch]))
            y = torch.from_numpy(np.stack([z[t:t+horizon] for t, _, _ in batch]))
            target = torch.from_numpy(np.stack([valid[t:t+horizon] for t, _, _ in batch]))
            prediction = model(x, mask, ctx)
            if begin == 0:
                if not torch.equal(prediction, model(torch.where(mask > 0, x, x+1234), mask, ctx)):
                    raise AssertionError("Hidden input value leaked into prediction")
            numerator = ((prediction-y).abs() * scale * target).sum(dim=(1, 2))
            denominator = target.sum(dim=(1, 2))
            if torch.any(denominator <= 0):
                raise AssertionError("Unmeasurable requests must be filtered uniformly")
            losses.append((numerator/denominator).numpy())
    return np.concatenate(losses).astype(np.float64)


def evaluate_events(model, events, groups, z, valid, context, scale, history, horizon, steps, equal_duration):
    # Deduplicate masks/origins independently of targets and outcomes.
    requests, indexes = [], {}

    def key_request(t, nodes, ages):
        key = (t, tuple(nodes), tuple(ages))
        if key not in indexes:
            indexes[key] = len(requests)
            requests.append(key)
        return indexes[key]

    plans, mechanisms = [], []
    for event in events:
        group = groups[event["group"]]
        nodes = group["nodes"]
        durations = [4, 4, 4] if equal_duration else group["durations_steps"]
        mechanism = []
        if event["mechanism_usable"]:
            for subset in range(8):
                mechanism.append(key_request(event["origin"], nodes, [2 if subset & (1 << i) else 0 for i in range(3)]))
        mechanisms.append(mechanism)
        event_plans = []
        for order in ORDERS:
            completion = np.zeros(3, dtype=int)
            clock = 0
            for station in order:
                clock += durations[station]
                completion[station] = clock
            trajectory = []
            for step in event["usable_steps"]:
                # First post-repair reading is timestamp origin+completion.
                # Inputs end at t-1, hence age is step-completion (no +1).
                ages = np.clip(step-completion, 0, history).tolist()
                t = event["origin"]+step
                joint = key_request(t, nodes, ages)
                outage = key_request(t, nodes, [0, 0, 0])
                singles = [key_request(t, nodes, [age if i == j else 0 for j in range(3)]) for i, age in enumerate(ages)]
                trajectory.append((joint, outage, singles))
                # Physical time check: no reading from before completion opened.
                for i, age in enumerate(ages):
                    if age and t-age < event["origin"]+completion[i]:
                        raise AssertionError("Pre-repair history reopened")
            event_plans.append(trajectory)
        plans.append(event_plans)
    losses = infer(model, requests, z, valid, context, scale, history, horizon)
    joint = np.empty((len(events), 6))
    additive = np.empty_like(joint)
    outage_cost = np.empty(len(events))
    for i, event_plans in enumerate(plans):
        for j, trajectory in enumerate(event_plans):
            actual, approximation, baseline = [], [], []
            for shared, empty, singles in trajectory:
                actual.append(losses[shared])
                baseline.append(losses[empty])
                approximation.append(sum(losses[k] for k in singles)-2*losses[empty])
            joint[i, j] = 5*sum(actual)
            additive[i, j] = 5*sum(approximation)
            if j == 0:
                outage_cost[i] = 5*sum(baseline)
    fixed = np.array([losses[m] if m else np.full(8, np.nan) for m in mechanisms])
    return joint, additive, outage_cost, fixed, len(requests)


def calibrated_policies(events, groups, joint, additive, fixed, equal_duration):
    policies = {}
    for group in groups:
        for hour in (6, 12, 18):
            eligible = [i for i, e in enumerate(events) if e["group"] == group["id"] and e["hour"] == hour]
            fallback = not eligible
            if fallback:
                eligible = [i for i, e in enumerate(events) if e["group"] == group["id"]]
            subset_events = [events[i] for i in eligible]
            jc, ac = day_macro(joint[eligible], subset_events), day_macro(additive[eligible], subset_events)
            durations = np.array([4, 4, 4] if equal_duration else group["durations_steps"])
            gains = fixed[eligible, 0:1]-fixed[eligible][:, [1, 2, 4]]
            measurable = np.isfinite(gains).all(axis=1)
            if not measurable.any():
                raise ValueError("No calibration cases for gain/processing-time policy")
            weights = day_macro(gains[measurable], [e for e, ok in zip(subset_events, measurable) if ok])
            ratio_order = tuple(sorted(range(3), key=lambda i: (-weights[i]/durations[i], i)))
            spt = tuple(sorted(range(3), key=lambda i: (durations[i], i)))
            policies[f"{group['id']}:{hour}"] = {
                "joint": int(np.argmin(jc)), "additive": int(np.argmin(ac)),
                "gain_per_time": ORDERS.index(ratio_order), "shortest_time": ORDERS.index(spt),
                "calibration_cases": len(eligible), "fallback_all_hours": fallback,
                "joint_costs": jc.tolist(), "additive_costs": ac.tolist(), "single_station_gains": weights.tolist()}
    return policies


def mechanism_summary(fixed, events):
    keep = np.isfinite(fixed).all(axis=1)
    loss = fixed[keep]
    measurable_events = [e for e, ok in zip(events, keep) if ok]
    gain = loss[:, 0:1]-loss
    interactions = np.stack([gain[:, (1 << i) | (1 << j)]-gain[:, 1 << i]-gain[:, 1 << j]
                             for i, j in itertools.combinations(range(3), 2)], axis=1)
    pair_masks = [3, 5, 6]
    approximation = np.stack([gain[:, 1 << i]+gain[:, 1 << j] for i, j in itertools.combinations(range(3), 2)], axis=1)
    estimated_choice = np.argmax(approximation, axis=1)
    true_choice = np.argmax(gain[:, pair_masks], axis=1)
    rows = np.arange(len(loss))
    regret = loss[rows, np.array(pair_masks)[estimated_choice]]-loss[rows, np.array(pair_masks)[true_choice]]
    full_residual = gain[:, 7]-gain[:, [1, 2, 4]].sum(axis=1)
    avg, interval = day_interval(regret, measurable_events)
    return {"cases": len(loss), "mean_signed_pair_interaction": float(day_macro(interactions.mean(axis=1), measurable_events)),
            "mean_absolute_pair_interaction": float(day_macro(np.abs(interactions).mean(axis=1), measurable_events)),
            "max_absolute_pair_interaction": float(np.abs(interactions).max()),
            "pair_choice_difference_fraction": float(day_macro((estimated_choice != true_choice).astype(float), measurable_events)),
            "pair_choice_actual_regret": float(avg), "pair_choice_actual_regret_ci": interval.tolist(),
            "mean_absolute_full_additivity_error": float(day_macro(np.abs(full_residual), measurable_events))}


def summarize_policy(events, actual, additive, outage, policies, fixed):
    choices = {name: np.array([policies[f"{e['group']}:{e['hour']}"][name] for e in events])
               for name in ("shortest_time", "gain_per_time", "additive", "joint")}
    choices["offline_additive_oracle"] = np.argmin(additive, axis=1)
    choices["offline_joint_oracle"] = np.argmin(actual, axis=1)
    rows = np.arange(len(events))
    costs = {name: actual[rows, order] for name, order in choices.items()}
    means = {name: float(day_macro(value, events)) for name, value in costs.items()}
    differences = costs["additive"]-costs["joint"]
    mean, interval = day_interval(differences, events)
    denominator = means["additive"]
    oracle_diff = costs["offline_additive_oracle"]-costs["offline_joint_oracle"]
    oracle_mean, oracle_ci = day_interval(oracle_diff, events)
    if np.any(oracle_diff < -1e-8):
        raise AssertionError("Exhaustive oracle cost must be minimal")
    result = {"events": len(events), "days": sorted({e['day'] for e in events}),
              "mean_evaluated_steps": float(np.mean([len(e['usable_steps']) for e in events])),
              "cumulative_loss_by_policy": means,
              "ongoing_outage_cumulative_loss": float(day_macro(outage, events)),
              "joint_vs_additive_improvement": float(mean), "joint_vs_additive_improvement_ci": interval.tolist(),
              "joint_vs_additive_relative_improvement": float(mean/denominator),
              "joint_vs_additive_relative_ci": relative_day_interval(differences, costs["additive"], events).tolist(),
              "calibrated_order_difference_fraction": float(day_macro((choices['additive'] != choices['joint']).astype(float), events)),
              "cases_joint_better": int(np.sum(differences > 1e-8)), "cases_joint_worse": int(np.sum(differences < -1e-8)),
              "offline_joint_vs_offline_additive_regret": float(oracle_mean), "offline_regret_ci": oracle_ci.tolist(),
              "offline_order_difference_fraction": float(day_macro((choices['offline_additive_oracle'] != choices['offline_joint_oracle']).astype(float), events)),
              "mechanism": mechanism_summary(fixed, events)}
    result["by_category"] = {}
    for category in sorted({e['category'] for e in events}):
        indexes = [i for i, e in enumerate(events) if e['category'] == category]
        es = [events[i] for i in indexes]
        delta, ci = day_interval(differences[indexes], es)
        result["by_category"][category] = {
            "events": len(es), "joint_vs_additive_improvement": float(delta), "ci": ci.tolist(),
            "relative_improvement": float(delta/day_macro(costs['additive'][indexes], es)),
            "mechanism": mechanism_summary(fixed[indexes], es)}
    return result, choices, costs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"outputs/multifault-pilot")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    data_path = ROOT/"data/raw/metr-la.h5"
    original = json.loads((ROOT/"outputs/recovery-pilot/config.json").read_text(encoding="utf-8"))
    z, valid, scale, mean, adjacency, timestamps, context, cuts, ids, columns = load_data(data_path, 42, 24, 1701, .9)
    if ids != original["selected_sensor_ids"] or not np.allclose(mean, original["normalization_mean"]):
        raise AssertionError("Archived input configuration changed")
    groups = choose_groups(adjacency)
    for g in groups:
        g["sensor_ids"] = [ids[i] for i in g["nodes"]]
    steps, history, horizon = 24, 12, 3
    events = {name: make_events(timestamps, valid, cuts, history, horizon, groups, part, steps)
              for name, part in (("calibration", 1), ("test", 2))}
    weights = {variant: {str(seed): sha256(ROOT/"outputs"/folder/f"model_seed{seed}.pt") for seed in (42, 43, 44)}
               for variant, folder in (("residual", "recovery-pilot"), ("no_residual", "recovery-pilot-no-residual"))}
    config = {"created_date": "2026-10-10", "protocol_sha256": sha256(ROOT/"docs/multifault_protocol.zh-CN.md"),
              "data_sha256": sha256(data_path), "weights_sha256": weights, "history": history, "horizon": horizon,
              "forecast_origins_steps": steps, "integration_minutes_per_step": 5, "group_selection_seed": 1010,
              "groups": groups, "events": events, "orders_local_node_indexes": ORDERS,
              "models_frozen": True, "policy_calibration_partition": "validation only",
              "decision_inputs": "known group, starting hour, synthetic durations; NO current/future test readings or labels",
              "research_screen_relative_gain": .03, "independent_confirmatory_evidence": False,
              "primary_metric": "day-macro cumulative network forecast MAE over 120min; residual model heterogeneous processing times",
              "replication": "no-residual model; equal-duration scenario secondary"}
    config_path = args.output/"config.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != json.loads(json.dumps(config)):
        raise ValueError("Frozen configuration differs; create another output directory")
    write_json(config_path, config)
    if args.prepare_only:
        print(json.dumps({"prepared": str(config_path), "group_count": len(groups), "events": {k: len(v) for k, v in events.items()}}, indent=2))
        return
    result = {"config": config, "variants": {}, "checks": {"mask_value_invariance": True, "no_precompletion_history": True}}
    csv_rows = []
    scale_t = torch.tensor(scale).view(1, 1, -1)
    for variant, folder in (("residual", "recovery-pilot"), ("no_residual", "recovery-pilot-no-residual")):
        result["variants"][variant] = {}
        for equal in (False, True):
            scenario = "equal_duration" if equal else "heterogeneous_duration"
            collected = {name: [] for name in events}
            for seed in (42, 43, 44):
                model = RecoveryForecaster(history, horizon, adjacency, variant == "residual")
                checkpoint = torch.load(ROOT/"outputs"/folder/f"model_seed{seed}.pt", weights_only=True)
                model.load_state_dict(checkpoint["state_dict"])
                model.eval()
                for partition, es in events.items():
                    values = evaluate_events(model, es, groups, z, valid, context, scale_t, history, horizon, steps, equal)
                    collected[partition].append(values[:4])
                    print(f"{variant} {scenario} seed={seed} {partition} events={len(es)} unique_inputs={values[4]}", flush=True)
            combined = {name: [np.stack([seed[i] for seed in values]).mean(axis=0) for i in range(4)]
                        for name, values in collected.items()}
            calibration = combined["calibration"]
            policies = calibrated_policies(events["calibration"], groups, calibration[0], calibration[1], calibration[3], equal)
            # Persist the policy before processing test costs/labels into reports.
            write_json(args.output/f"policies_{variant}_{scenario}.json", policies)
            actual, additive, outage, fixed = combined["test"]
            summary, choices, costs = summarize_policy(events["test"], actual, additive, outage, policies, fixed)
            per_seed = []
            for seed, values in zip((42, 43, 44), collected["test"]):
                delta = values[0][np.arange(len(actual)), choices["additive"]]-values[0][np.arange(len(actual)), choices["joint"]]
                per_seed.append({"seed": seed, "same_frozen_policy_improvement": float(day_macro(delta, events["test"]))})
            summary["per_seed_same_policy"] = per_seed
            result["variants"][variant][scenario] = summary
            for i, event in enumerate(events["test"]):
                csv_rows.append({"variant": variant, "scenario": scenario, "day": event["day"], "hour": event["hour"],
                                 "group": event["group"], "category": event["category"], "evaluated_steps": len(event["usable_steps"]),
                                 **{f"order_{k}": int(v[i]) for k, v in choices.items()}, **{f"loss_{k}": float(v[i]) for k, v in costs.items()},
                                 "joint_minus_additive_loss": float(costs["joint"][i]-costs["additive"][i])})
            # Exact additivity for structurally disjoint outputs, at every mask/time, within float noise.
            disjoint = [i for i, e in enumerate(events["test"]) if e["category"] == "disjoint_outputs"]
            maximum = float(np.max(np.abs(actual[disjoint]-additive[disjoint])))
            summary["disjoint_max_cumulative_additivity_error"] = maximum
            if maximum > .005:
                raise AssertionError(f"Disjoint outputs must be additive; max difference {maximum}")
            # Raw numeric arrays make all six orders and counterfactual masks auditable.
            np.savez_compressed(args.output/f"arrays_{variant}_{scenario}.npz",
                                actual=actual, additive=additive, outage=outage, fixed=fixed,
                                calibration_actual=calibration[0], calibration_additive=calibration[1], calibration_fixed=calibration[3])
            print(json.dumps({"variant": variant, "scenario": scenario, "relative_gain": summary["joint_vs_additive_relative_improvement"],
                              "relative_ci": summary["joint_vs_additive_relative_ci"], "mechanism": summary["mechanism"]}, indent=2), flush=True)
    primary = result["variants"]["residual"]["heterogeneous_duration"]
    replication = result["variants"]["no_residual"]["heterogeneous_duration"]
    result["passes_exploratory_research_screen"] = bool(primary["joint_vs_additive_relative_improvement"] >= .03
          and primary["joint_vs_additive_relative_ci"][0] > 0 and replication["joint_vs_additive_relative_improvement"] > 0)
    result["checks"]["disjoint_output_additivity"] = True
    result["elapsed_seconds"] = time.perf_counter()-started
    write_json(args.output/"results.json", result)
    with (args.output/"episode_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Done: elapsed={result['elapsed_seconds']:.1f}s exploratory_screen={result['passes_exploratory_research_screen']}", flush=True)


if __name__ == "__main__":
    main()
