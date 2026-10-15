"""Evaluate regularly sampled event alerts using only the Python standard library.

This generic protocol is separate from the frozen PEMS-BAY thesis protocol.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import html
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class Sample:
    station_id: str
    timestamp: datetime
    score: float | None
    is_event: int | None


def load_csv(path: str | Path) -> list[Sample]:
    """Read timestamp, station_id, score, is_event; blank score/label means unknown."""
    rows = []
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"station_id", "timestamp", "score", "is_event"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("CSV requires station_id,timestamp,score,is_event")
        for line, row in enumerate(reader, 2):
            try:
                if None in row or any(row[k] is None for k in required):
                    raise ValueError("malformed CSV row")
                station = row["station_id"].strip()
                if not station:
                    raise ValueError("empty station_id")
                instant = datetime.fromisoformat(row["timestamp"].strip().replace("Z", "+00:00"))
                if instant.utcoffset() is None:
                    raise ValueError("timestamp requires a UTC offset, e.g. Z or +08:00")
                score_text = row["score"].strip()
                score = float(score_text) if score_text else None
                if score is not None and (not math.isfinite(score) or not 0 <= score <= 1):
                    raise ValueError("score must be finite and between 0 and 1, or blank")
                label = row["is_event"].strip()
                if label not in {"", "0", "1"}:
                    raise ValueError("is_event must be 0, 1, or blank")
                rows.append(Sample(station, instant.astimezone(timezone.utc), score,
                                   int(label) if label else None))
            except (ValueError, TypeError) as exc:
                raise ValueError(f"CSV line {line}: {exc}") from exc
    if not rows:
        raise ValueError("CSV has no samples")
    return rows


def evaluate(rows: list[Sample], *, threshold: float = 0.5, step_minutes: int = 5,
             horizon_minutes: int = 15, early_minutes: int = 10) -> dict:
    """Return auditable event matches and eligible decision exposure.

    Scores must be available at their row timestamp. Labels are evaluation-only.
    Thresholds must be fixed on a separate calibration split before evaluation.
    """
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be finite and within [0, 1]")
    if (not isinstance(step_minutes, int) or step_minutes <= 0
            or not isinstance(horizon_minutes, int) or horizon_minutes < 0
            or horizon_minutes % step_minutes
            or not isinstance(early_minutes, int) or not 0 <= early_minutes <= horizon_minutes):
        raise ValueError("positive integer step; horizon is a nonnegative step multiple; 0 <= early <= horizon")
    if not rows:
        raise ValueError("no samples")
    ordered = sorted(rows, key=lambda row: (row.station_id, row.timestamp))
    segments, segment, previous = [], [], None
    gaps = unknown_labels = missing_scores = 0
    for row in ordered:
        if (not row.station_id or row.timestamp.utcoffset() is None
                or row.is_event not in (None, 0, 1)
                or (row.score is not None and (not math.isfinite(row.score) or not 0 <= row.score <= 1))):
            raise ValueError("invalid Sample; use load_csv for CSV validation")
        same = previous is not None and row.station_id == previous.station_id
        delta = (row.timestamp - previous.timestamp).total_seconds() if same else None
        if same and (delta <= 0 or delta % (step_minutes * 60)):
            raise ValueError("duplicate timestamps or timestamps off the station sampling grid")
        gap = same and delta != step_minutes * 60
        gaps += int(gap)
        if not same or gap or row.is_event is None:
            if segment:
                segments.append(segment)
                segment = []
        unknown_labels += int(row.is_event is None)
        missing_scores += int(row.score is None)
        if row.is_event is not None:
            segment.append(row)
        previous = row
    if segment:
        segments.append(segment)

    horizon = horizon_minutes // step_minutes
    events, alerts, excluded_runs = [], [], []
    eligible_ticks = false_alarms = duplicate_alerts = excluded_alerts = 0
    for part in segments:
        n = len(part)
        runs, i = [], 0
        while i < n:
            if part[i].is_event == 0:
                i += 1
                continue
            start = i
            while i + 1 < n and part[i + 1].is_event == 1:
                i += 1
            end = i
            complete = start > 0 and end < n - 1 and start >= horizon and end + horizon < n
            run = {"start": start, "end": end, "eligible": complete, "event": None}
            if complete:
                run["event"] = {"station_id": part[0].station_id,
                    "onset": part[start].timestamp.isoformat(), "end": part[end].timestamp.isoformat(),
                    "first_alert": None, "lead_minutes": None}
                events.append(run["event"])
            else:
                excluded_runs.append({"station_id": part[0].station_id,
                    "onset": part[start].timestamp.isoformat(), "end": part[end].timestamp.isoformat(),
                    "reason": "boundary event or insufficient known horizon context"})
            runs.append(run)
            i += 1

        # Unknown truth and gaps split segments. Ineligible event windows are not negatives.
        eligible = [i + horizon < n and not any(
            not run["eligible"] and run["start"] - horizon <= i <= run["end"]
            for run in runs) for i in range(n)]
        eligible_ticks += sum(eligible)
        active = None
        for i, row in enumerate(part):
            if row.score is None:
                active = None
                continue
            positive = row.score >= threshold
            if not positive:
                active = False
                continue
            if active is True:
                continue
            left_censored = active is None
            active = True
            if left_censored or not eligible[i]:
                excluded_alerts += 1
                alerts.append({"station_id": row.station_id, "timestamp": row.timestamp.isoformat(),
                               "status": "excluded", "reason": "unknown episode start or ineligible context"})
                continue
            candidates = [run for run in runs if run["eligible"]
                          and run["start"] - horizon <= i <= run["end"]]
            if not candidates:
                false_alarms += 1
                status = "false_alarm"
                matched = None
            else:
                # Earliest compatible event owns the alert, including repeat alerts.
                matched = candidates[0]["event"]
                if matched["first_alert"] is None:
                    matched["first_alert"] = row.timestamp.isoformat()
                    matched["lead_minutes"] = (part[candidates[0]["start"]].timestamp - row.timestamp).total_seconds() / 60
                    status = "matched"
                else:
                    duplicate_alerts += 1
                    status = "duplicate"
            alerts.append({"station_id": row.station_id, "timestamp": row.timestamp.isoformat(),
                           "status": status, "event_onset": matched["onset"] if matched else None})

    detected = sum(event["first_alert"] is not None for event in events)
    early = sum(event["lead_minutes"] is not None and event["lead_minutes"] >= early_minutes for event in events)
    exposure = eligible_ticks * step_minutes / 1440
    return {"protocol": "traffic-alert-csv-v1", "synthetic": False,
        "settings": {"threshold": threshold, "step_minutes": step_minutes,
                     "horizon_minutes": horizon_minutes, "early_minutes": early_minutes},
        "metrics": {"eligible_events": len(events), "detected_events": detected, "early_events": early,
                    "event_detection_rate": detected / len(events) if events else None,
                    "early_event_recall": early / len(events) if events else None,
                    "false_alarm_episodes": false_alarms, "duplicate_alert_episodes": duplicate_alerts,
                    "eligible_station_days": exposure,
                    "false_alarms_per_eligible_station_day": false_alarms / exposure if exposure else None},
        "audit": {"rows": len(rows), "stations": len({row.station_id for row in rows}),
                  "known_segments": len(segments), "sampling_gaps": gaps,
                  "unknown_label_rows": unknown_labels, "missing_score_rows": missing_scores,
                  "excluded_event_runs": len(excluded_runs), "excluded_alert_episodes": excluded_alerts,
                  "eligible_decision_ticks": eligible_ticks},
        "events": events, "alerts": alerts, "excluded_event_runs": excluded_runs,
        "limitations": ["Evaluation-only labels must never be model inputs.",
            "Use a threshold frozen on separate calibration data; this tool does not select it.",
            "A sustained positive episode can match only its earliest compatible event.",
            "Generic timestamp lead has no thesis-specific +one-step offset.",
            "Rates pool event counts and eligible exposure across stations; zero denominators return null."]}


def render_html(result: dict) -> str:
    """Self-contained report; escape all strings originating from user data."""
    esc = lambda value: html.escape(str(value), quote=True)
    metrics = "".join(f"<tr><th>{esc(key)}</th><td>{esc(value)}</td></tr>" for key, value in result["metrics"].items())
    events = "".join("<tr>" + "".join(f"<td>{esc(event[key])}</td>" for key in
                     ("station_id", "onset", "end", "first_alert", "lead_minutes")) + "</tr>" for event in result["events"])
    notes = "".join(f"<li>{esc(note)}</li>" for note in result["limitations"])
    return ("<!doctype html><html lang='en'><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            "<title>Traffic Alert Lab | CSV report</title><style>"
            "body{font:16px system-ui;background:#f3f6fa;color:#17283d;max-width:1080px;margin:40px auto;padding:0 20px}"
            "table{border-collapse:collapse;background:white;width:100%;margin:24px 0}"
            "th,td{text-align:left;border-bottom:1px solid #dce5ef;padding:12px;overflow-wrap:anywhere}"
            ".scroll{overflow:auto}h1{font-size:34px}code{overflow-wrap:anywhere}</style>"
            f"<h1>Traffic Alert Lab</h1><p>{'SYNTHETIC TEACHING EXAMPLE' if result['synthetic'] else 'User CSV evaluation'}"
            " · traffic-alert-csv-v1</p><p>Event detection, early warning and false alarms must be read together.</p>"
            f"<p>Settings: <code>{esc(json.dumps(result['settings']))}</code></p><table>{metrics}</table>"
            "<h2>Event matches</h2><div class='scroll'><table><tr><th>Station</th><th>Onset</th><th>End</th>"
            f"<th>First alert</th><th>Lead / min</th></tr>{events}</table></div>"
            f"<h2>Audit</h2><p><code>{esc(json.dumps(result['audit']))}</code></p><ul>{notes}</ul>"
            "<p>Complete alert classifications and exclusions are saved in report.json.</p></html>")


def write_report(result: dict, output: str | Path) -> None:
    output = Path(output)
    # A new directory avoids overwriting a previous evaluation or user files.
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    (output / "report.html").write_text(render_html(result), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="New report directory")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--step-minutes", type=int, default=5)
    parser.add_argument("--horizon-minutes", type=int, default=15)
    parser.add_argument("--early-minutes", type=int, default=10)
    parser.add_argument("--synthetic", action="store_true", help="Label the report as synthetic teaching data")
    args = parser.parse_args()
    try:
        result = evaluate(load_csv(args.csv), threshold=args.threshold, step_minutes=args.step_minutes,
                          horizon_minutes=args.horizon_minutes, early_minutes=args.early_minutes)
        result["synthetic"] = args.synthetic
        result["input_sha256"] = hashlib.sha256(args.csv.read_bytes()).hexdigest()
        write_report(result, args.output)
    except (ValueError, FileExistsError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
