#!/usr/bin/env python3
"""Market-clock helpers shared by the remaining-trend daily pipeline."""

from __future__ import annotations

import pandas as pd


NYSE_TIMEZONE = "America/New_York"


def current_nyse_timestamp(now=None) -> pd.Timestamp:
    """Return a timezone-aware current timestamp in New York."""
    if now is None:
        return pd.Timestamp.now(tz=NYSE_TIMEZONE)
    ts = pd.Timestamp(now)
    if ts.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    return ts.tz_convert(NYSE_TIMEZONE)


def current_nyse_trade_date(now=None) -> pd.Timestamp:
    """Return the New York calendar date as a naive normalized Timestamp.

    ``now`` is injectable for tests. When supplied it must be timezone-aware so
    that converting across the Shanghai/New York date boundary is unambiguous.
    """
    ts = current_nyse_timestamp(now)
    return ts.tz_localize(None).normalize()
