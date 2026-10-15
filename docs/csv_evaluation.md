# Evaluate your own alert scores

[中文说明](csv_evaluation.zh-CN.md) · [Tiny training tutorial](tiny_training.zh-CN.md)

The generic evaluator needs **Python 3.12 and no third-party packages**. It accepts scores from any model. It does not train a model or choose a threshold.

```bash
python alert_eval.py examples/synthetic_alerts.csv --threshold 0.5 --synthetic --output runs/example-report
```

Open `runs/example-report/report.html`. `report.json` contains every event match, alert classification, exclusion and the input SHA-256. Use a **new output directory** for each run. The supplied CSV is entirely synthetic: two events, one detected 15 minutes early, one missed event, and one false alert.

For your own data, omit `--synthetic`:

```bash
python alert_eval.py my_scores.csv --step-minutes 5 --horizon-minutes 15 --early-minutes 10 --threshold 0.5 --output runs/my-report
```

## Browser entry

[Online lab](https://lanausee-debug.github.io/traffic-alert-lab/#online-lab): choose or paste CSV, adjust exploratory settings and download reports. No CSV upload to a server. Browser limits: 2 MiB, 20,000 rows, whole-second timestamps with timezone (no fractional seconds); strict CSV quoting, unique headers and matching column counts. The Python CLI accepts a broader ISO timestamp syntax. UTF-8 file BOM is removed when decoding, and the browser report SHA hashes the decoded text as UTF-8, not necessarily the original file bytes. Python CLI SHA hashes the original bytes.

## CSV contract

```csv
station_id,timestamp,score,is_event
A,2020-01-01T00:00:00Z,0.10,0
A,2020-01-01T00:05:00Z,0.80,0
A,2020-01-01T00:10:00Z,0.30,1
```

This three-row snippet shows syntax, not a complete evaluable event.

| Column | Meaning |
| --- | --- |
| `station_id` | Nonempty station identifier |
| `timestamp` | **Decision/score availability time**, ISO 8601 with timezone; converted to UTC |
| `score` | Risk score in [0, 1]; blank means unavailable |
| `is_event` | Evaluation-only truth at this timestamp: 0, 1, or blank for unknown |

Rows are sorted per station. Duplicate timestamps and times off the declared station sampling grid are rejected. Gaps split evaluation segments; the tool does not interpolate or silently treat unknown labels as negatives. Choose the physical event definition upstream and document it; speed thresholds are not inferred by this tool.

Scores must use only information available at their timestamp. A forecast's target time is not automatically its availability time. Select the threshold and event rules on separate training/calibration data **before** evaluating. This CLI deliberately has no threshold optimization on the evaluation CSV.

## Exact protocol: `traffic-alert-csv-v1`

1. A known segment consists of consecutive, regularly spaced rows with nonblank truth. Unknown truth, gaps and station changes start a new segment.
2. An event is a maximal run of truth=1. It is eligible only when known zeros bracket both ends, at least one full anticipation horizon precedes its onset, and a full horizon follows its final positive tick within the known segment. All other event runs and their anticipation windows are excluded, not scored as negative events.
3. An alert is a threshold crossing into a positive episode (`score >= threshold`). Sustained positives produce one alert. At segment starts or immediately after missing scores, an initial positive episode has an unknown start and is excluded until a known negative resets it. Missing scores are not imputed.
4. Eligible alert starts require the full forward horizon to be known and must be outside excluded event windows. Each alert belongs to the **earliest** eligible event whose onset is no more than the horizon after the alert, or which is currently active. Repeated alerts for that event are reported as duplicates; they never detect an additional event. A long positive episode does not automatically detect later events.
5. An eligible alert with no compatible event is a false alarm. Lead is `event onset timestamp - alert start timestamp` in minutes: positive is early, zero is at onset, negative is late. Detection includes late matches; early recall additionally requires lead >= the configured early requirement.
6. Eligible exposure sums one sampling interval for each eligible decision tick, regardless of score availability. False alarms per eligible station-day = false alert episodes / (eligible ticks × step minutes / 1440). This is **decision-time exposure**, not raw calendar coverage or the thesis denominator. Counts pool across stations; no station averaging or uncertainty interval is implied.

The minimum score at a segment boundary does not establish a pre-segment negative. These conservative rules deliberately discard ambiguous boundary data. Audit exclusions and score availability alongside the metrics: missing outputs can reduce false alarms while increasing missed events. Zero denominators produce JSON `null`, never a fabricated zero rate.

**This is a new generic utility protocol, not the frozen thesis replay.** Its timestamp lead has no thesis-specific +one-step offset, its boundary rules differ, and its metrics must not be substituted for the published PEMS-BAY table. Statistical confidence intervals and train/calibration split validation are not implemented here; split provenance remains the caller's responsibility.

## Python API

```python
from alert_eval import load_csv, evaluate, write_report

result = evaluate(load_csv("my_scores.csv"), threshold=0.5,
                  step_minutes=5, horizon_minutes=15, early_minutes=10)
write_report(result, "runs/api-report")
```

Run the hand-calculated boundary checks with `python -m unittest discover -s tests -p test_csv_alerts.py -v`.
