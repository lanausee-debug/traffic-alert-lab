"""Hand-checked cases for public CSV evaluation semantics, no PyTorch needed."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import unittest

from alert_eval import Sample, evaluate, load_csv, render_html, write_report


def timeline(length=40, event_ranges=((10, 12), (25, 27)), alarms=(7, 20), station="A"):
    origin = datetime(2020, 1, 1, tzinfo=timezone.utc)
    return [Sample(station, origin + timedelta(minutes=5 * i), 0.9 if i in alarms else 0.1,
                   int(any(start <= i <= end for start, end in event_ranges))) for i in range(length)]


class CsvAlertsTest(unittest.TestCase):
    def test_hand_calculated_event_and_exposure(self):
        result = evaluate(timeline())
        metrics = result["metrics"]
        self.assertEqual(metrics["eligible_events"], 2)
        self.assertEqual(metrics["detected_events"], 1)
        self.assertEqual(metrics["early_events"], 1)
        self.assertEqual(metrics["false_alarm_episodes"], 1)
        self.assertAlmostEqual(metrics["eligible_station_days"], 37 * 5 / 1440)
        self.assertEqual(result["events"][0]["lead_minutes"], 15)
        self.assertEqual(result["events"][1]["first_alert"], None)

    def test_late_alert_and_duplicates(self):
        result = evaluate(timeline(alarms=(11, 13)))
        self.assertEqual(result["events"][0]["lead_minutes"], -5)
        self.assertEqual(result["metrics"]["early_events"], 0)
        self.assertEqual(result["metrics"]["false_alarm_episodes"], 1)
        result = evaluate(timeline(alarms=(7, 9)))
        self.assertEqual(result["metrics"]["duplicate_alert_episodes"], 1)
        self.assertEqual(result["metrics"]["false_alarm_episodes"], 0)

    def test_one_episode_cannot_detect_two_events(self):
        rows = timeline(event_ranges=((10, 12), (15, 17)), alarms=tuple(range(8, 19)))
        result = evaluate(rows)
        self.assertEqual(result["metrics"]["detected_events"], 1)
        self.assertEqual(len([a for a in result["alerts"] if a["status"] == "matched"]), 1)

    def test_gaps_and_unknown_labels_censor_events(self):
        rows = timeline()
        result = evaluate([row for i, row in enumerate(rows) if i != 11])
        self.assertEqual(result["audit"]["sampling_gaps"], 1)
        self.assertEqual(result["metrics"]["eligible_events"], 1)
        rows[11] = Sample("A", rows[11].timestamp, rows[11].score, None)
        result = evaluate(rows)
        self.assertEqual(result["audit"]["unknown_label_rows"], 1)
        self.assertEqual(result["metrics"]["eligible_events"], 1)
        self.assertEqual(result["metrics"]["false_alarm_episodes"], 1)

    def test_edge_events_and_unknown_alarm_start(self):
        result = evaluate(timeline(event_ranges=((0, 2), (37, 39)), alarms=(0, 35)))
        self.assertEqual(result["metrics"]["eligible_events"], 0)
        self.assertEqual(result["audit"]["excluded_event_runs"], 2)
        self.assertEqual(result["metrics"]["false_alarm_episodes"], 0)
        rows = timeline(alarms=(7, 8))
        rows[6] = Sample("A", rows[6].timestamp, None, 0)
        result = evaluate(rows)
        self.assertEqual(result["metrics"]["detected_events"], 0)
        self.assertEqual(result["audit"]["excluded_alert_episodes"], 1)

    def test_exact_lead_and_threshold_boundaries(self):
        result = evaluate(timeline(alarms=(8,)), threshold=0.9)
        self.assertEqual(result["metrics"]["early_events"], 1)
        self.assertEqual(result["events"][0]["lead_minutes"], 10)
        result = evaluate(timeline(alarms=(9,)))
        self.assertEqual(result["metrics"]["early_events"], 0)

    def test_stations_are_independent(self):
        result = evaluate(timeline() + timeline(station="B", alarms=(22,)))
        self.assertEqual(result["metrics"]["eligible_events"], 4)
        self.assertEqual(result["metrics"]["detected_events"], 2)
        self.assertEqual(result["audit"]["stations"], 2)

    def test_bad_grid_settings_and_null_rates(self):
        rows = timeline()
        with self.assertRaises(ValueError):
            evaluate(rows + [rows[0]])
        rows[1] = Sample("A", rows[1].timestamp + timedelta(seconds=1), 0.1, 0)
        with self.assertRaises(ValueError):
            evaluate(rows)
        for settings in ({"threshold": float("nan")}, {"step_minutes": 0},
                         {"horizon_minutes": 7}, {"early_minutes": 20}):
            with self.assertRaises(ValueError):
                evaluate(timeline(), **settings)
        result = evaluate(timeline(length=2, event_ranges=(), alarms=()))
        self.assertIsNone(result["metrics"]["event_detection_rate"])
        self.assertIsNone(result["metrics"]["false_alarms_per_eligible_station_day"])

    def test_csv_input_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.csv"
            header = "station_id,timestamp,score,is_event\n"
            for bad in ("A,2020-01-01T00:00:00,0.4,0", "A,2020-01-01T00:00:00Z,nan,0",
                        "A,2020-01-01T00:00:00Z,0.4,unknown", "A,2020-01-01T00:00:00Z,1.2,0"):
                path.write_text(header + bad, encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_csv(path)
            path.write_text(header + "A,2020-01-01T08:00:00+08:00,,\n", encoding="utf-8")
            row = load_csv(path)[0]
            self.assertEqual(row.timestamp.hour, 0)
            self.assertIsNone(row.score)
            self.assertIsNone(row.is_event)

    def test_report_escapes_user_strings_and_refuses_overwrite(self):
        result = evaluate(timeline(station="<script>alert(1)</script>"))
        self.assertNotIn("<script>", render_html(result))
        self.assertIn("&lt;script&gt;", render_html(result))
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "report"
            write_report(result, destination)
            self.assertTrue((destination / "report.json").is_file())
            with self.assertRaises(FileExistsError):
                write_report(result, destination)


if __name__ == "__main__":
    unittest.main()
