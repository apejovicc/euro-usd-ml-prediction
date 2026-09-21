from __future__ import annotations
import os
import pandas as pd


def fetch_eurusd_prices(start_date: str, end_date: str) -> pd.DataFrame:
    import yfinance as yf
    raw = yf.download(
        "EURUSD=X", start=start_date, end=end_date,
        progress=False, auto_adjust=False,
    )
    if raw.empty:
        raise RuntimeError(
            "yfinance nije vratio podatke - provjerite internet konekciju."
        )
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw[["Open", "High", "Low", "Close"]].copy()
    df.index.name = "Date"
    return df


def reconstruct_point_in_time_series(fred_client, series_id: str,
                                      start_date: str, end_date: str) -> pd.Series:

    releases = fred_client.get_series_all_releases(series_id)
    releases = releases.sort_values("realtime_start").reset_index(drop=True)

    is_new_period = releases["date"] == releases["date"].cummax()
    current_value = releases["value"].where(is_new_period).ffill()

    timeline_df = pd.DataFrame({
        "realtime_start": releases["realtime_start"],
        "value": current_value,
    }).drop_duplicates("realtime_start", keep="last")

    timeline_df = timeline_df.set_index("realtime_start").sort_index()
    timeline_df.index = pd.to_datetime(timeline_df.index)

    daily_index = pd.bdate_range(start_date, end_date)
    combined_index = timeline_df.index.union(daily_index)
    pit_series = timeline_df["value"].reindex(combined_index).sort_index().ffill()
    return pit_series.reindex(daily_index).astype(float)

def fetch_macro_data(fred_api_key: str, start_date: str, end_date: str,
                      series_lag_days: dict, vintage_series: list) -> pd.DataFrame:
    import fredapi
    fred = fredapi.Fred(api_key=fred_api_key)

    daily_index = pd.bdate_range(start_date, end_date)
    columns = {}

    for series_id in series_lag_days:
        if series_id in vintage_series:
            columns[series_id] = reconstruct_point_in_time_series(
                fred, series_id, start_date, end_date
            )
        else:
            s = fred.get_series(series_id, observation_start=start_date, observation_end=end_date)
            columns[series_id] = s.reindex(daily_index).ffill()

    macro = pd.DataFrame(columns)
    return macro

def build_raw_dataset(fred_api_key: str, cfg: dict, save_path: str | None = None) -> pd.DataFrame:
    start_date = cfg["data"]["start_date"]
    end_date = cfg["data"]["end_date"]

    prices = fetch_eurusd_prices(start_date, end_date)
    macro = fetch_macro_data(
        fred_api_key, start_date, end_date,
        cfg["fred"]["series_lag_days"], cfg["fred"]["vintage_series"],
    )
    raw = prices.join(macro, how="left").ffill().dropna()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        raw.to_csv(save_path)

    return raw