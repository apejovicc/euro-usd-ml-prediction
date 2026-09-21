from __future__ import annotations
import os
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

FEATURE_COLUMNS = [
    "Return", "SMA_ratio", "Volatility_5D", "Momentum_10D",
    "Daily_Range", "RSI_14", "CPI_YoY", "UNRATE_Diff", "Rate_Differential_Change",
]


def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()

    rs = pd.Series(np.nan, index=close.index, dtype=float)
    both_zero = (avg_loss == 0) & (avg_gain == 0)
    loss_zero_gain_positive = (avg_loss == 0) & (avg_gain > 0)
    rs.loc[loss_zero_gain_positive] = np.inf
    rs.loc[both_zero] = 1.0
    rs.loc[~both_zero & ~loss_zero_gain_positive] = avg_gain / avg_loss

    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def build_feature_matrix(raw: pd.DataFrame, save_path: str | None = None) -> tuple[pd.DataFrame, list[str]]:
    if raw.empty:
        raise ValueError("raw DataFrame mora imati bar jedan red.")

    df = raw.copy().sort_index()
    close = df["Close"]
    n = len(df)
    momentum_period = max(1, min(10, n - 1))

    df["Return"] = close.pct_change()
    sma5 = close.rolling(5).mean()
    sma20 = close.rolling(20).mean()
    df["SMA_ratio"] = sma5 / sma20
    df["Volatility_5D"] = df["Return"].rolling(5).std()
    df["Momentum_10D"] = close.pct_change(periods=momentum_period)
    df["Daily_Range"] = (df["High"] - df["Low"]) / close
    df["RSI_14"] = compute_rsi(close, 14)

    df["CPI_YoY"] = df["CPIAUCSL"].pct_change(periods=252)
    df["UNRATE_Diff"] = df["UNRATE"].diff(periods=21)
    df["Rate_Differential_Change"] = df["DFF"].diff() - df["ECBDFR"].diff()

    df["target"] = (close.shift(-1) > close).astype(float)
    df.loc[df.index[-1], "target"] = np.nan  # posljednja opservacija - nema Close(t+1)

    model_df = df[FEATURE_COLUMNS + ["target"]].dropna().copy()
    if model_df.empty:
        raise ValueError(
            "build_feature_matrix() je nakon konstrukcije karakteristika i dropna() dobio prazan DataFrame. "
            "Povecajte broj uzoraka ili provjerite ulazne serije."
        )
    model_df["target"] = model_df["target"].astype(int)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        model_df.to_csv(save_path)

    return model_df, FEATURE_COLUMNS


def dataset_summary(model_df: pd.DataFrame, feature_cols: list[str]) -> dict:
    return {
        "n_samples": len(model_df),
        "date_start": str(model_df.index.min().date()),
        "date_end": str(model_df.index.max().date()),
        "class_balance": model_df["target"].value_counts(normalize=True).to_dict(),
        "feature_columns": feature_cols,
    }


def check_stationarity(model_df: pd.DataFrame, feature_cols: list[str],
                        significance: float = 0.05, save_path: str | None = None) -> pd.DataFrame:
    rows = []
    for col in feature_cols:
        series = model_df[col].dropna()
        stat, pvalue, used_lag, nobs, crit_values, _ = adfuller(series, autolag="AIC")
        rows.append({
            "feature": col,
            "adf_statistic": round(stat, 4),
            "p_value": round(pvalue, 6),
            "n_obs": nobs,
            "used_lag": used_lag,
            "critical_1%": round(crit_values["1%"], 4),
            "critical_5%": round(crit_values["5%"], 4),
            "critical_10%": round(crit_values["10%"], 4),
            "stationary_at_5%": bool(pvalue < significance),
        })
    result = pd.DataFrame(rows).set_index("feature")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        result.to_csv(save_path)

    return result