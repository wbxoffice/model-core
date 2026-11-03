from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from market_time_utils import current_nyse_trade_date
from merge_spx_5min_sources_incremental import _write_prediction_input
import run_trend_modes_daily as daily
from run_trend_modes_daily import (
    _append_results,
    _is_30min_source,
    _resolve_prediction_intraday_source,
    _resolve_run_mode,
    _select_prediction_trade_date,
    _validate_open90_completeness,
)
from backfill_trend_mode_predictions import _candidate_dates


def _bar(ts: str, source: str, close: float) -> dict:
    return {
        "datetime": pd.Timestamp(ts),
        "open": close,
        "high": close + 1,
        "low": close - 1,
        "close": close,
        "volume": 100,
        "source": source,
    }


class DailyPipelineRuntimeTests(unittest.TestCase):
    def test_new_york_trade_date_across_shanghai_date_boundary(self):
        self.assertEqual(
            current_nyse_trade_date(pd.Timestamp("2026-01-02 00:30", tz="Asia/Shanghai")),
            pd.Timestamp("2026-01-01"),
        )
        self.assertEqual(
            current_nyse_trade_date(pd.Timestamp("2026-08-27 00:30", tz="Asia/Shanghai")),
            pd.Timestamp("2026-08-26"),
        )

    def test_prediction_input_contains_history_and_current_session(self):
        history = pd.DataFrame([_bar("2026-08-26 13:00", "vdtr_spx_5min", 100.0)])
        current = pd.DataFrame(
            [
                _bar("2026-08-27 06:30", "vdtr_spx_5min", 101.0),
                _bar("2026-08-27 06:30", "exported_latest", 102.0),
                _bar("2026-08-27 07:55", "exported_latest", 103.0),
            ]
        )
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "prediction.csv"
            combined = _write_prediction_input(
                history,
                [current],
                out,
                {"vdtr_spx_5min": 1, "exported_latest": 4},
            )
            saved = pd.read_csv(out)

        self.assertEqual(len(combined), 3)
        self.assertEqual(len(saved), 3)
        selected = combined.loc[combined["datetime"] == pd.Timestamp("2026-08-27 06:30")].iloc[0]
        self.assertEqual(selected["source"], "exported_latest")
        self.assertEqual(float(selected["close"]), 102.0)

    def test_30min_source_is_detected_but_rejected_for_daily_prediction(self):
        self.assertTrue(_is_30min_source(Path("SPCFD_SPX, 30_abcd.csv")))
        self.assertFalse(_is_30min_source(Path("SPCFD_SPX, 5_abcd.csv")))
        with patch.object(daily, "_discover_intraday_file", return_value=Path("SPCFD_SPX, 30_abcd.csv")):
            with self.assertRaisesRegex(ValueError, "requires a 5min"):
                _resolve_prediction_intraday_source()

    def test_backfill_uses_only_missing_posttrain_nyse_dates(self):
        frame = pd.DataFrame(
            {
                "target_date": pd.to_datetime(["2026-07-02", "2026-07-03", "2026-07-06", "2026-07-07"]),
                "real_trend_remaining_after_open90": [0, 0, 1, 0],
                "real_trend_close150_after_open90": [0, 0, 0, 0],
            }
        )
        dates = _candidate_dates(
            frame=frame,
            existing_dates={pd.Timestamp("2026-07-07")},
            open_dates={pd.Timestamp("2026-07-02"), pd.Timestamp("2026-07-06"), pd.Timestamp("2026-07-07")},
            train_end=pd.Timestamp("2026-07-02"),
        )
        self.assertEqual(dates, [pd.Timestamp("2026-07-06")])

    def test_open90_requires_exact_18_five_minute_bars(self):
        trade_date = pd.Timestamp("2026-08-27")
        expected = pd.date_range("2026-08-27 06:30", periods=18, freq="5min")
        with tempfile.TemporaryDirectory() as td:
            good = Path(td) / "SPX_5min_prediction_input.csv"
            pd.DataFrame({"time": [*expected, pd.Timestamp("2026-08-27 12:55")]}).to_csv(good, index=False)
            status = _validate_open90_completeness(good, trade_date)
            self.assertTrue(status["complete"])
            self.assertEqual(status["actual_bars"], 18)
            self.assertEqual(_resolve_run_mode("auto", status), "postclose")

            bad = Path(td) / "SPX_5min_incomplete.csv"
            pd.DataFrame({"time": expected.delete(5)}).to_csv(bad, index=False)
            with self.assertRaisesRegex(ValueError, "missing=\\['06:55'\\]"):
                _validate_open90_completeness(bad, trade_date)

            no_today = Path(td) / "SPX_5min_no_today.csv"
            pd.DataFrame({"time": [pd.Timestamp("2026-08-26 12:55")]}).to_csv(no_today, index=False)
            with self.assertRaisesRegex(ValueError, "expected=18 actual=0"):
                _validate_open90_completeness(no_today, trade_date)

    def test_direct_python_refreshes_even_when_prepared_file_already_exists(self):
        with tempfile.TemporaryDirectory() as td:
            raw = Path(td) / "SPX_5min_raw.csv"
            prepared = Path(td) / "SPX_5min_prediction_input.csv"
            pd.DataFrame({"time": [pd.Timestamp("2026-08-27 07:55")]}).to_csv(raw, index=False)
            pd.DataFrame({"time": [pd.Timestamp("2026-08-27 07:55")]}).to_csv(prepared, index=False)
            with (
                patch.object(daily, "_discover_intraday_file", return_value=raw),
                patch.object(daily, "PREPARED_5M_INPUT", prepared),
                patch.object(daily.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as run,
            ):
                resolved = _resolve_prediction_intraday_source(refresh_5m=True)

        self.assertEqual(resolved, prepared)
        run.assert_called_once()

    def test_ny_preopen_uses_latest_completed_source_date_but_after_open_requires_today(self):
        previous = pd.date_range("2026-08-27 06:30", periods=79, freq="5min")
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "SPX_5min_prediction_input.csv"
            pd.DataFrame({"time": previous}).to_csv(source, index=False)

            target, reason = _select_prediction_trade_date(
                source,
                pd.Timestamp("2026-08-28 00:30", tz="America/New_York"),
            )
            after_open_target, after_open_reason = _select_prediction_trade_date(
                source,
                pd.Timestamp("2026-08-28 11:05", tz="America/New_York"),
            )

        self.assertEqual(target, pd.Timestamp("2026-08-27"))
        self.assertEqual(reason, "ny_preopen_latest_source_date")
        self.assertEqual(after_open_target, pd.Timestamp("2026-08-28"))
        self.assertEqual(after_open_reason, "current_date_required")

    def test_intraday_and_postclose_rows_coexist_but_same_stage_replaces(self):
        trade_date = current_nyse_trade_date().date().isoformat()

        def row(run_mode: str, probability: float) -> pd.DataFrame:
            return pd.DataFrame(
                [
                    {
                        "date": trade_date,
                        "label": "real_trend_remaining_after_open90",
                        "mode": "balanced",
                        "run_mode": run_mode,
                        "threshold": 0.27,
                        "p_trend": probability,
                        "trend_pred": int(probability >= 0.27),
                    }
                ]
            )

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "predictions.csv"
            _append_results(row("intraday", 0.30), out)
            _append_results(row("postclose", 0.31), out)
            _append_results(row("intraday", 0.40), out)
            saved = pd.read_csv(out)

        self.assertEqual(len(saved), 2)
        self.assertEqual(set(saved["run_mode"]), {"intraday", "postclose"})
        intraday = saved[saved["run_mode"] == "intraday"].iloc[0]
        self.assertAlmostEqual(float(intraday["p_trend"]), 0.40)


if __name__ == "__main__":
    unittest.main()
