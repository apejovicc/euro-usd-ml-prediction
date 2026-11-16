"""
Preuzimanje sirovih podataka.

- EUR/USD dnevne OHLC cijene: Yahoo Finance (yfinance).
- Makroekonomski indikatori: FRED, uz ALFRED "point-in-time" rekonstrukciju
  za CPI i nezaposlenost (sprjecava curenje informacija iz buducnosti,
  ukljucujuci naknadne revizije podataka, ne samo kasnjenje objave).

TRENUTAK PREDVIDJANJA (definicija koja vazi za cijeli projekat):
  Predvidjanje za dan t se donosi NAKON zatvaranja trgovanja na dan t,
  kada je poznata cijena zatvaranja Close(t), a cilj je smjer Close(t+1)
  u odnosu na Close(t). Skup informacija raspolozivih u tom trenutku:
    - cijene (Open, High, Low, Close) zakljucno sa danom t;
    - CPIAUCSL i UNRATE: posljednja vrijednost cija je objava
      (realtime_start iz ALFRED-a) <= t;
    - DFF i ECBDFR: vrijednost za dan t - lag (lag = 1 radni dan), jer se
      vrijednost DFF za radni dan d objavljuje tek ujutro narednog radnog
      dana (NY Fed objavljuje EFFR za prethodni radni dan oko 9:00 h), pa
      vrijednost za dan t nije poznata u trenutku predvidjanja na dan t.

FRED API kljuc se NIKAD ne cuva u kodu ni u config.yaml - prosljedjuje se
kao argument (u notebook-u pribavljen preko getpass).
"""
from __future__ import annotations
import os
import pandas as pd


def fetch_eurusd_prices(start_date: str, end_date: str) -> pd.DataFrame:
    """Preuzima dnevne OHLC cijene za EUR/USD sa Yahoo Finance."""
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
    """Rekonstruise 'point-in-time' seriju: za svaki dan t, vrijednost koja
    je bila STVARNO poznata tog dana (koristeci realtime_start iz ALFRED-a),
    a ne finalna/revidirana vrijednost koju FRED danas prikazuje.
    """
    releases = fred_client.get_series_all_releases(series_id)
    releases = releases.sort_values("realtime_start")

    max_period = None
    current_value = None
    timeline = []
    for _, row in releases.iterrows():
        if max_period is None or row["date"] >= max_period:
            max_period = row["date"]
            current_value = row["value"]
        timeline.append({"realtime_start": row["realtime_start"], "value": current_value})

    timeline_df = pd.DataFrame(timeline).drop_duplicates("realtime_start", keep="last")
    timeline_df = timeline_df.set_index("realtime_start").sort_index()
    timeline_df.index = pd.to_datetime(timeline_df.index)

    daily_index = pd.bdate_range(start_date, end_date)
    combined_index = timeline_df.index.union(daily_index)
    pit_series = timeline_df["value"].reindex(combined_index).sort_index().ffill()
    return pit_series.reindex(daily_index).astype(float)


def fetch_macro_data(fred_api_key: str, start_date: str, end_date: str,
                      series_lag_days: dict, vintage_series: list) -> pd.DataFrame:
    """Gradi makro DataFrame.

    - Serije iz `vintage_series` (CPIAUCSL, UNRATE): ALFRED point-in-time
      rekonstrukcija. Kasnjenje objave je VEC sadrzano u datumima objave
      (realtime_start), pa se vrijednost iz `series_lag_days` za njih
      IGNORISE (da se kasnjenje ne bi primijenilo dvaput).
    - Ostale (dnevne) serije (DFF, ECBDFR): vrijednost za datum d
      pomjera se za `series_lag_days[serie]` RADNIH dana unaprijed, tj.
      na dan t koristi se vrijednost za dan t - lag. Za DFF je lag=1 jer se
      efektivna stopa za radni dan d objavljuje ujutro narednog radnog dana.
    """
    import fredapi
    fred = fredapi.Fred(api_key=fred_api_key)

    daily_index = pd.bdate_range(start_date, end_date)
    columns = {}

    for series_id, lag in series_lag_days.items():
        if series_id in vintage_series:
            columns[series_id] = reconstruct_point_in_time_series(
                fred, series_id, start_date, end_date
            )
        else:
            if lag < 0:
                raise ValueError(f"series_lag_days[{series_id}] ne smije biti negativan.")
            s = fred.get_series(series_id, observation_start=start_date, observation_end=end_date)
            s = s.reindex(daily_index).ffill()
            columns[series_id] = s.shift(lag) if lag > 0 else s

    macro = pd.DataFrame(columns)
    return macro


def build_raw_dataset(fred_api_key: str, cfg: dict, save_path: str | None = None) -> pd.DataFrame:
    """Preuzima cijene + makro podatke, spaja ih, i (opciono) cuva u
    data/raw/. Ovo je jedina funkcija koju notebook/pipeline poziva
    direktno za korak preuzimanja podataka."""
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
    