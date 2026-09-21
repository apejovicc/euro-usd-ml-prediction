from __future__ import annotations
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


def chronological_split(model_df: pd.DataFrame, train_fraction: float,
                         val_fraction: float, gap: int = 1):
 
    if model_df.empty:
        raise ValueError("model_df mora imati bar jedan red.")
    if not 0 <= train_fraction < 1:
        raise ValueError("train_fraction mora biti u [0, 1).")
    if not 0 <= val_fraction < 1:
        raise ValueError("val_fraction mora biti u [0, 1).")
    if train_fraction + val_fraction >= 1:
        raise ValueError("train_fraction + val_fraction mora biti < 1.")
    if gap < 0:
        raise ValueError("gap mora biti >= 0.")

    n = len(model_df)
    n_train = int(n * train_fraction)
    n_val = int(n * val_fraction)
    n_test = n - n_train - n_val

    if n_train < 1 or n_test < 1:
        raise ValueError(
            "Nedovoljno uzoraka za validnu hronolosku podjelu. "
            f"Potreban je barem 1 red u trening i test skupu; dobiveno n={n}, "
            f"n_train={n_train}, n_val={n_val}, n_test={n_test}."
        )
    if gap >= n_train or (n_val > 0 and gap >= n_val):
        raise ValueError(
            "gap je prevelik za datu podjelu: mora biti manji od duzine train i val skupa."
        )

    train_df_full = model_df.iloc[:n_train]
    val_df_full = model_df.iloc[n_train:n_train + n_val]
    test_df = model_df.iloc[n_train + n_val:]

    train_df = train_df_full.iloc[:-gap] if gap > 0 else train_df_full
    val_df = val_df_full.iloc[:-gap] if gap > 0 else val_df_full

    if train_df.empty or test_df.empty:
        raise ValueError("Purging je rezultirao praznim skupom. Povećajte broj uzoraka ili smanjite gap.")

    split_info = {
        name: {
            "n": len(d),
            "start": str(d.index.min().date()),
            "end": str(d.index.max().date()),
        }
        for name, d in [("train", train_df), ("val", val_df), ("test", test_df)]
    }
    split_info["gap_days"] = gap
    split_info["purged_rows"] = {
        "train": len(train_df_full) - len(train_df),
        "val": len(val_df_full) - len(val_df),
    }
    return train_df, val_df, test_df, split_info


def describe_cv_folds(train_df: pd.DataFrame, n_splits: int, gap: int = 1) -> list[dict]:

    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    fold_info = []
    for i, (tr_idx, val_idx) in enumerate(tscv.split(train_df), start=1):
        fold_info.append({
            "fold": i,
            "train_n": len(tr_idx),
            "train_start": str(train_df.index[tr_idx[0]].date()),
            "train_end": str(train_df.index[tr_idx[-1]].date()),
            "gap_days": gap,
            "val_n": len(val_idx),
            "val_start": str(train_df.index[val_idx[0]].date()),
            "val_end": str(train_df.index[val_idx[-1]].date()),
        })
    return fold_info


def make_xy(df: pd.DataFrame, feature_cols: list[str]):
    return df[feature_cols].values, df["target"].values