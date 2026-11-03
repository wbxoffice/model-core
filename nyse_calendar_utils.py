#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    GoodFriday,
    Holiday,
    USLaborDay,
    USMartinLutherKingJr,
    USMemorialDay,
    USPresidentsDay,
    USThanksgivingDay,
    nearest_workday,
)


class _NYSEHolidayCalendar(AbstractHolidayCalendar):
    rules = [
        Holiday("NewYearsDay", month=1, day=1, observance=nearest_workday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday(
            "JuneteenthNationalIndependenceDay",
            month=6,
            day=19,
            start_date="2022-01-01",
            observance=nearest_workday,
        ),
        Holiday("IndependenceDay", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("ChristmasDay", month=12, day=25, observance=nearest_workday),
    ]


def _build_nyse_open_days(start_date: str, end_date: str) -> pd.DatetimeIndex:
    start = pd.to_datetime(start_date).normalize()
    end = pd.to_datetime(end_date).normalize()
    weekdays = pd.bdate_range(start=start, end=end, freq="C")
    holidays = _NYSEHolidayCalendar().holidays(start=start, end=end).normalize()
    return weekdays.difference(holidays)


def generate_nyse_open_days_csv(
    csv_path: Path | str,
    start_year: int = 1990,
    end_year: int | None = None,
) -> Path:
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if end_year is None:
        end_year = datetime.now().year + 1

    open_days = _build_nyse_open_days(
        start_date=f"{start_year}-01-01",
        end_date=f"{end_year}-12-31",
    )
    df = pd.DataFrame({"date": open_days})
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["weekday"] = df["date"].dt.day_name()
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def ensure_nyse_open_days_csv(
    csv_path: Path | str,
    start_year: int = 1990,
    ensure_through_year: int | None = None,
) -> Path:
    path = Path(csv_path)
    if ensure_through_year is None:
        ensure_through_year = datetime.now().year + 1

    if not path.exists():
        return generate_nyse_open_days_csv(path, start_year=start_year, end_year=ensure_through_year)

    try:
        old = pd.read_csv(path, usecols=["year"])
        max_year = int(old["year"].max()) if not old.empty else 0
    except Exception:
        max_year = 0

    if max_year < ensure_through_year:
        return generate_nyse_open_days_csv(path, start_year=start_year, end_year=ensure_through_year)
    return path


def load_nyse_open_days(csv_path: Path | str) -> pd.DatetimeIndex:
    df = pd.read_csv(csv_path, usecols=["date"])
    idx = pd.to_datetime(df["date"], errors="coerce").dropna().dt.normalize()
    return pd.DatetimeIndex(sorted(idx.unique()))


def next_nyse_open_day(date_value, open_days: pd.DatetimeIndex) -> pd.Timestamp:
    d = pd.to_datetime(date_value).normalize()
    pos = open_days.searchsorted(d, side="left")
    if pos >= len(open_days):
        raise ValueError(f"No NYSE open day available on/after {d.date()} in loaded calendar.")
    return pd.Timestamp(open_days[pos]).normalize()


def align_series_to_next_open_day(series: pd.Series, open_days: pd.DatetimeIndex) -> pd.Series:
    s = pd.to_datetime(series, errors="coerce").dt.normalize()
    return s.apply(lambda x: next_nyse_open_day(x, open_days) if pd.notna(x) else pd.NaT)

