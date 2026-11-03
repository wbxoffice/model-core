#!/user/bin/env python3
# -*- coding: utf-8 -*-
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Build event_calendar.csv for SPX VDTR v2.0

- 自动抓取 2012–2024 年 FOMC 决议日（第二天）
- 手动补充 2025–2026 年 FOMC 暂定决议日
- 生成:
    is_fomc, is_fomc_window, days_to_next_fomc, days_from_prev_fomc
    is_vix_expiry, days_to_vix_expiry
    is_spy_opex_m, days_to_spy_opex_m
- 保存到 data/event_calendar.csv
"""

from pathlib import Path
import re

import pandas as pd
import requests
from bs4 import BeautifulSoup
from pandas.tseries.offsets import BDay


# =========================================================
# 1. 抓取 2012–2024 年 FOMC 日期（来自 Fed 官方历史页面）
# =========================================================

HISTORICAL_URL_TEMPLATE = (
    "https://www.federalreserve.gov/monetarypolicy/fomchistorical{year}.htm"
)

MONTH_MAP = {
    # January
    "January": 1, "Jan": 1, "Jan.": 1,
    # February
    "February": 2, "Feb": 2, "Feb.": 2,
    # March
    "March": 3, "Mar": 3, "Mar.": 3,
    # April
    "April": 4, "Apr": 4, "Apr.": 4,
    # May
    "May": 5,
    # June
    "June": 6, "Jun": 6, "Jun.": 6,
    # July
    "July": 7, "Jul": 7, "Jul.": 7,
    # August
    "August": 8, "Aug": 8, "Aug.": 8,
    # September
    "September": 9, "Sep": 9, "Sep.": 9, "Sept": 9, "Sept.": 9,
    # October
    "October": 10, "Oct": 10, "Oct.": 10,
    # November
    "November": 11, "Nov": 11, "Nov.": 11,
    # December
    "December": 12, "Dec": 12, "Dec.": 12,
}


def parse_meeting_header(text: str):
    """
    解析类似:
      'January 24-25 Meeting - 2012'
      'March 13 Meeting - 2012'
      'April/May 30-1 Meeting - 2013'
      'Jan. 31-Feb. 1 Meeting - 2017'

    返回决议日日期 (datetime64) 或 None
    """

    text = text.strip()

    # 正则：Month(/Month2) Day1(-Day2) Meeting - Year
    m = re.match(
        r"^([A-Za-z\.]+(?:/[A-Za-z\.]+)?)\s+(\d{1,2})(?:[-–](\d{1,2}))?\s+Meeting\s+-\s+(\d{4})",
        text,
    )
    if not m:
        return None

    month_token_raw, d1_str, d2_str, year_str = m.groups()
    year = int(year_str)
    day1 = int(d1_str)
    day2 = int(d2_str) if d2_str is not None else None

    # 先去掉所有小数点，比如 'Jan.' -> 'Jan'
    month_token_raw = month_token_raw.replace(".", "")

    # 处理跨月写法，如 'April/May' 或 'Jan/Feb'
    if "/" in month_token_raw:
        month1_str, month2_str = [s.strip() for s in month_token_raw.split("/")]
        # 声明日 = 第二天 -> 取第二个月份 + 第二个数字
        if month2_str not in MONTH_MAP:
            return None
        month = MONTH_MAP[month2_str]
        day = day2 if day2 is not None else day1
    else:
        month_str = month_token_raw.strip()
        if month_str not in MONTH_MAP:
            return None
        month = MONTH_MAP[month_str]
        # 区间 -> 取第二天；单日 -> 用该日
        day = day2 if day2 is not None else day1

    try:
        return pd.Timestamp(year=year, month=month, day=day)
    except ValueError:
        # 防止异常日期，比如解析失败
        return None


def fetch_fomc_dates_2012_2019():
    """只抓 2012–2019，使用 old 'fomchistoricalYYYY.htm' 页面。"""
    all_dates = []

    for year in range(2012, 2020):  # 只到 2019，避免 2020 404
        url = HISTORICAL_URL_TEMPLATE.format(year=year)
        print(f"Fetching {year} FOMC page: {url}")
        resp = requests.get(url)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        headers = soup.find_all(["h4", "h5"])
        for h in headers:
            date_obj = parse_meeting_header(h.get_text())
            if date_obj is not None:
                all_dates.append(date_obj)

    all_dates = sorted(set(all_dates))
    return pd.to_datetime(all_dates)

def get_fomc_dates_2020_2024():
    """
    2020–2024 FOMC 决议日（含 2020 年几次重要非常规会议）。
    日期均来源于 Fed 官方 meeting calendars 页面。
    """

    # 2020: 正常会 + 两次紧急会议 + 3/23 特别声明
    dates_2020 = [
        "2020-01-29",
        "2020-03-03",  # unscheduled
        "2020-03-15",  # unscheduled
        "2020-03-23",  # notation vote + 重要声明
        "2020-04-29",
        "2020-06-10",
        "2020-07-29",
        "2020-09-16",
        "2020-11-05",
        "2020-12-16",
    ]

    # 2021: 8 次常规会议
    dates_2021 = [
        "2021-01-27",
        "2021-03-17",
        "2021-04-28",
        "2021-06-16",
        "2021-07-28",
        "2021-09-22",
        "2021-11-03",
        "2021-12-15",
    ]

    # 2022
    dates_2022 = [
        "2022-01-26",
        "2022-03-16",
        "2022-05-04",
        "2022-06-15",
        "2022-07-27",
        "2022-09-21",
        "2022-11-02",
        "2022-12-14",
    ]

    # 2023
    dates_2023 = [
        "2023-02-01",
        "2023-03-22",
        "2023-05-03",
        "2023-06-14",
        "2023-07-26",
        "2023-09-20",
        "2023-11-01",
        "2023-12-13",
    ]

    # 2024（截至 Fed 已公布的 8 次会议）
    dates_2024 = [
        "2024-01-31",
        "2024-03-20",
        "2024-05-01",
        "2024-06-12",
        "2024-07-31",
        "2024-09-18",
        "2024-11-07",
        "2024-12-18",
    ]

    return pd.to_datetime(dates_2020 + dates_2021 + dates_2022 + dates_2023 + dates_2024)

# =========================================================
# 2. 补上 2025–2026 年的暂定 FOMC 日期
#    （来自官方公布的 meeting schedule）
# =========================================================

def get_future_fomc_dates_2025_2026():
    # 都写成 "YYYY-MM-DD"（第二天）
    dates_2025 = [
        "2025-01-29",
        "2025-03-19",
        "2025-05-07",
        "2025-06-18",
        "2025-07-30",
        "2025-09-17",
        "2025-10-29",
        "2025-12-10",
    ]
    dates_2026 = [
        "2026-01-28",
        "2026-03-18",
        "2026-04-29",
        "2026-06-17",
        "2026-07-29",
        "2026-09-16",
        "2026-10-28",
        "2026-12-09",
    ]
    return pd.to_datetime(dates_2025 + dates_2026)


def get_all_fomc_dates():
    """
    综合 2012–2019 的抓取结果 + 2020–2024 的硬编码 + 2025–2026 暂定日程。
    """
    past = fetch_fomc_dates_2012_2019()
    mid = get_fomc_dates_2020_2024()
    future = get_future_fomc_dates_2025_2026()

    all_dates = sorted(set(past.tolist() + mid.tolist() + future.tolist()))
    return pd.to_datetime(all_dates)



# =========================================================
# 3. SPY 月度 OPEX（每月第 3 个周五）
# =========================================================

def monthly_opex(start_year=2012, end_year=2026):
    opex_dates = []
    for year in range(start_year, end_year + 1):
        days = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="B")
        fridays = days[days.weekday == 4]  # 周五
        # 每个月的第 3 个周五
        by_month = fridays.to_series().groupby(fridays.month)
        for _, series in by_month:
            if len(series) >= 3:
                opex_dates.append(series.iloc[2])
    return pd.to_datetime(sorted(opex_dates))


# =========================================================
# 4. VIX 到期日（简化：每月第 3 个周三）
# =========================================================

def vix_expiry_dates(start_year=2012, end_year=2026):
    vix_dates = []
    for year in range(start_year, end_year + 1):
        days = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="B")
        weds = days[days.weekday == 2]  # 周三
        by_month = weds.to_series().groupby(weds.month)
        for _, series in by_month:
            if len(series) >= 3:
                vix_dates.append(series.iloc[2])
    return pd.to_datetime(sorted(vix_dates))


# =========================================================
# 5. 生成 event_calendar DataFrame
# =========================================================

def build_event_calendar(start="2012-01-01", end="2026-12-31") -> pd.DataFrame:
    date_index = pd.date_range(start, end, freq="B")
    df = pd.DataFrame({"date": date_index})

    # ---------- FOMC ----------
    fomc_dates = get_all_fomc_dates()
    df["is_fomc"] = df["date"].isin(fomc_dates).astype(int)

    # FOMC window: 决议前后 ±2 个交易日
    df["is_fomc_window"] = 0
    for d in fomc_dates:
        window = pd.date_range(d - BDay(2), d + BDay(2), freq="B")
        df.loc[df["date"].isin(window), "is_fomc_window"] = 1

    # days_to_next_fomc / days_from_prev_fomc （按自然日差）
    df["days_to_next_fomc"] = 999
    df["days_from_prev_fomc"] = 999

    for i, cur in df["date"].items():
        # next
        future = fomc_dates[fomc_dates >= cur]
        if len(future) > 0:
            df.at[i, "days_to_next_fomc"] = (future.min() - cur).days
        # prev
        past = fomc_dates[fomc_dates <= cur]
        if len(past) > 0:
            df.at[i, "days_from_prev_fomc"] = (cur - past.max()).days

    # ---------- SPY 月度 OPEX ----------
    opex_dates = monthly_opex(start_year=2012, end_year=2026)
    df["is_spy_opex_m"] = df["date"].isin(opex_dates).astype(int)

    df["days_to_spy_opex_m"] = 999
    for i, cur in df["date"].items():
        future = opex_dates[opex_dates >= cur]
        if len(future) > 0:
            df.at[i, "days_to_spy_opex_m"] = (future.min() - cur).days

    # ---------- VIX 到期 ----------
    vix_dates = vix_expiry_dates(start_year=2012, end_year=2026)
    df["is_vix_expiry"] = df["date"].isin(vix_dates).astype(int)

    df["days_to_vix_expiry"] = 999
    for i, cur in df["date"].items():
        future = vix_dates[vix_dates >= cur]
        if len(future) > 0:
            df.at[i, "days_to_vix_expiry"] = (future.min() - cur).days

    return df


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    calendar = build_event_calendar()
    out_path = data_dir / "event_calendar.csv"
    calendar.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Saved event calendar to: {out_path}")
