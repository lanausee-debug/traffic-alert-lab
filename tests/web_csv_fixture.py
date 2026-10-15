"""Independent Python reference outputs for browser/Node protocol parity."""
import csv
from datetime import datetime, timedelta, timezone
import io
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alert_eval import Sample, evaluate


def main():
    cases = []
    origin = datetime(2020, 1, 1, tzinfo=timezone.utc)
    for seed in range(60):
        rng = random.Random(seed)
        rows = []
        for station in ('A', 'B'):
            truth = 0
            for i in range(70):
                if rng.random() < 0.12:
                    truth = 1 - truth
                if rng.random() < 0.03:
                    continue
                label = None if rng.random() < 0.04 else truth
                score = None if rng.random() < 0.08 else rng.choice((0.1, 0.35, 0.5, 0.65, 0.8, 0.95))
                rows.append(Sample(station, origin + timedelta(minutes=5 * i), score, label))
        for settings in ({}, {'threshold': 0.8}, {'threshold': 0.35, 'horizon_minutes': 10, 'early_minutes': 5},
                         {'horizon_minutes': 0, 'early_minutes': 0}):
            buffer = io.StringIO(newline='')
            writer = csv.writer(buffer)
            writer.writerow(('station_id', 'timestamp', 'score', 'is_event'))
            writer.writerows((row.station_id, row.timestamp.isoformat(), row.score, row.is_event) for row in reversed(rows))
            cases.append({'name': f'seed-{seed}-{settings}', 'csv': buffer.getvalue(),
                          'settings': settings, 'expected': evaluate(rows, **settings)})
    # Quoting, Unicode order, timezone offsets, same-instant station independence.
    rows = [Sample(station, origin + timedelta(minutes=5*i), 0.9 if i in (7, 9, 21) else 0.1,
                   int(10 <= i <= 12 or 25 <= i <= 27))
            for station in ('站点,甲', '"B"', '\U00010000', '\ue000') for i in range(40)]
    buffer = io.StringIO(newline=''); writer = csv.writer(buffer)
    writer.writerow(('station_id', 'timestamp', 'score', 'is_event'))
    writer.writerows((row.station_id, row.timestamp.astimezone(timezone(timedelta(hours=8))).isoformat(),
                     row.score, row.is_event) for row in reversed(rows))
    cases.append({'name': 'quoted-unicode-timezone', 'csv': '\ufeff'+buffer.getvalue(),
                  'settings': {}, 'expected': evaluate(rows)})
    print(json.dumps(cases, ensure_ascii=True, allow_nan=False))


if __name__ == '__main__': main()
