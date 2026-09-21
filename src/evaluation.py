from __future__ import annotations
import os
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
)
from statsmodels.stats.contingency_tables import mcnemar

def majority_baseline(y_train: np.ndarray, y_test: np.ndarray) -> np.ndarray:
    clf = DummyClassifier(strategy="most_frequent")
    clf.fit(np.zeros((len(y_train), 1)), y_train)
    return clf.predict(np.zeros((len(y_test), 1)))

def stratified_random_baseline(y_train: np.ndarray, y_test: np.ndarray, seed: int) -> np.ndarray:
    clf = DummyClassifier(strategy="stratified", random_state=seed)
    clf.fit(np.zeros((len(y_train), 1)), y_train)
    return clf.predict(np.zeros((len(y_test), 1)))

def persistence_baseline(returns_test: np.ndarray) -> np.ndarray:
    return (returns_test > 0).astype(int)

def ar1_baseline(returns_train: np.ndarray, returns_train_next: np.ndarray,
                  returns_test: np.ndarray) -> np.ndarray:

    X_design = np.column_stack([np.ones(len(returns_train)), returns_train])
    beta, *_ = np.linalg.lstsq(X_design, returns_train_next, rcond=None)
    alpha, b = beta
    predicted_next_return = alpha + b * returns_test
    return (predicted_next_return > 0).astype(int)

def compute_all_baselines(y_trainval, y_test, returns_test, seed: int,
                           returns_trainval: np.ndarray | None = None,
                           returns_trainval_next: np.ndarray | None = None) -> dict:
    baselines = {
        "baseline_majority": majority_baseline(y_trainval, y_test),
        "baseline_random": stratified_random_baseline(y_trainval, y_test, seed),
        "baseline_persistence": persistence_baseline(returns_test),
    }
    if returns_trainval is not None and returns_trainval_next is not None:
        baselines["baseline_ar1"] = ar1_baseline(returns_trainval, returns_trainval_next, returns_test)
    return baselines


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "n_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }

def build_results_table(model_predictions: dict) -> pd.DataFrame: 
    rows = []
    for name, (y_true, y_pred) in model_predictions.items():
        m = compute_metrics(y_true, y_pred)
        rows.append({"model": name, **m})
    return pd.DataFrame(rows).set_index("model").sort_values("accuracy", ascending=False)

def bootstrap_accuracy_ci(y_true, y_pred, n_boot: int = 2000, seed: int = 42, alpha: float = 0.05) -> dict:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    correct = (np.asarray(y_true) == np.asarray(y_pred)).astype(float)

    idx = rng.integers(0, n, size=(n_boot, n))
    accs = correct[idx].mean(axis=1)

    return {
        "point_estimate": float(correct.mean()),
        "ci_lower": float(np.percentile(accs, 100 * alpha / 2)),
        "ci_upper": float(np.percentile(accs, 100 * (1 - alpha / 2))),
    }

def subperiod_accuracy(y_true, y_pred, dates, n_periods: int = 3) -> pd.DataFrame:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    edges = np.linspace(0, n, n_periods + 1, dtype=int)

    rows = []
    for i in range(n_periods):
        lo, hi = edges[i], edges[i + 1]
        acc = float((y_true[lo:hi] == y_pred[lo:hi]).mean())
        rows.append({
            "period": i + 1,
            "start": str(pd.Timestamp(dates[lo]).date()),
            "end": str(pd.Timestamp(dates[hi - 1]).date()),
            "n": hi - lo,
            "accuracy": round(acc, 4),
        })
    return pd.DataFrame(rows)

def mcnemar_test(y_true, y_pred_a, y_pred_b) -> dict:
    y_true, y_pred_a, y_pred_b = np.asarray(y_true), np.asarray(y_pred_a), np.asarray(y_pred_b)
    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)
    n_01 = int(np.sum((~correct_a) & correct_b))
    n_10 = int(np.sum(correct_a & (~correct_b)))
    result = mcnemar([[0, n_01], [n_10, 0]], exact=(n_01 + n_10 < 25))
    return {"statistic": float(result.statistic), "p_value": float(result.pvalue),
            "n_01": n_01, "n_10": n_10}

def save_predictions_csv(model_predictions: dict, path: str) -> None:
   
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = []
    for name, (y_true, y_pred) in model_predictions.items():
        for i, (yt, yp) in enumerate(zip(y_true, y_pred)):
            rows.append({"model": name, "index": i, "y_true": int(yt), "y_pred": int(yp)})
    pd.DataFrame(rows).to_csv(path, index=False)

def save_metrics_csv(results_table: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    results_table.to_csv(path)