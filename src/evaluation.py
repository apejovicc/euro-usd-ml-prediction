"""
Baseline modeli + evaluacija - jedinstven izvor istine za sve
tabele/matrice/testove (mentorska tacka 7).

Sve funkcije rade iskljucivo nad (y_true, y_pred) parovima - nista se
rucno ne prepisuje niti naknadno mijenja.
"""
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
    """y_hat(t) = 1[Return(t) > 0] - Return(t) je vec poznat u trenutku
    predvidjanja za dan t+1, nema curenja informacija."""
    return (returns_test > 0).astype(int)


def ar1_baseline(returns_train: np.ndarray, returns_train_next: np.ndarray,
                  returns_test: np.ndarray) -> np.ndarray:
    """Jednostavan autoregresivni (AR(1)) referentni model (mentorska
    tacka 8): Return(t+1) = alpha + beta*Return(t) + e, ocijenjen OLS-om
    ISKLJUCIVO na trening/trening+validacionom skupu. Klasa se dobija iz
    predznaka fitovane vrijednosti za Return(t) na testnom skupu. Za
    razliku od persistence_baseline (koji samo kopira predznak), ovaj
    model procjenjuje koeficijent momentuma/mean-reversion iz podataka.
    """
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
    """model_predictions: {model_name: (y_true, y_pred)}. Generise JEDNU
    tabelu za sve modele, direktno iz istih predikcija koje se koriste i
    za matrice konfuzije i za statisticke testove."""
    rows = []
    for name, (y_true, y_pred) in model_predictions.items():
        m = compute_metrics(y_true, y_pred)
        rows.append({"model": name, **m})
    return pd.DataFrame(rows).set_index("model").sort_values("accuracy", ascending=False)


# ---------------------------------------------------------------------
# Statisticka znacajnost (mentorska tacka 9)
# ---------------------------------------------------------------------

def bootstrap_accuracy_ci(y_true, y_pred, n_boot: int = 2000, seed: int = 42, alpha: float = 0.05) -> dict:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    correct = (np.asarray(y_true) == np.asarray(y_pred)).astype(float)
    accs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        accs[b] = correct[idx].mean()
    return {
        "point_estimate": float(correct.mean()),
        "ci_lower": float(np.percentile(accs, 100 * alpha / 2)),
        "ci_upper": float(np.percentile(accs, 100 * (1 - alpha / 2))),
    }


def subperiod_accuracy(y_true, y_pred, dates, n_periods: int = 3) -> pd.DataFrame:
    """Dijeli (y_true, y_pred, dates) hronoloski na n_periods jednakih
    dijelova i racuna tacnost u svakom - mentorska tacka 9/H3 ("prikazati
    ponasanje modela kroz vise odvojenih vremenskih perioda"). Uravnotezenija
    matrica konfuzije sama po sebi NIJE dokaz generalizacije - ova funkcija
    daje direktan, provjerljiv uvid u stabilnost kroz vrijeme."""
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
    n_01 = int(np.sum((~correct_a) & correct_b))   # A pogrijesio, B pogodio
    n_10 = int(np.sum(correct_a & (~correct_b)))   # A pogodio, B pogrijesio
    if n_01 + n_10 == 0:   # modeli se ne razlikuju ni u jednom primjeru
        return {"statistic": 0.0, "p_value": 1.0, "n_01": 0, "n_10": 0}
    result = mcnemar([[0, n_01], [n_10, 0]], exact=(n_01 + n_10 < 25))
    return {"statistic": float(result.statistic), "p_value": float(result.pvalue),
            "n_01": n_01, "n_10": n_10}


# ---------------------------------------------------------------------
# Dopunske statisticke procedure
# ---------------------------------------------------------------------

def block_bootstrap_accuracy_ci(y_true, y_pred, block_length: int = 10, n_boot: int = 2000,
                                seed: int = 42, alpha: float = 0.05) -> dict:
    """Interval pouzdanosti tacnosti pomocu bootstrapa sa pokretnim blokovima
    (moving block bootstrap). Za razliku od obicnog bootstrapa koji pretpostavlja
    nezavisne opservacije, blokovi uzastopnih dana cuvaju eventualnu serijsku
    zavisnost nizova tacnih/netacnih predikcija. Koristi se kao provjera
    osjetljivosti, ne kao zamjena za obican bootstrap."""
    rng = np.random.default_rng(seed)
    correct = (np.asarray(y_true) == np.asarray(y_pred)).astype(float)
    n = len(correct)
    L = int(max(1, min(block_length, n)))
    n_blocks = int(np.ceil(n / L))
    starts = rng.integers(0, n - L + 1, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(L)[None, None, :]).reshape(n_boot, -1)[:, :n]
    accs = correct[idx].mean(axis=1)
    return {
        "point_estimate": float(correct.mean()),
        "ci_lower": float(np.percentile(accs, 100 * alpha / 2)),
        "ci_upper": float(np.percentile(accs, 100 * (1 - alpha / 2))),
        "block_length": L,
    }


def random_baseline_distribution(y_train, y_test, n_draws: int = 10000, seed: int = 42) -> dict:
    """Raspodjela tacnosti stratifikovanog slucajnog baznog modela (predvidja
    klasu 1 sa vjerovatnocom p_train neovisno za svaki dan). Jedno izvlacenje
    (baseline_random) je samo jedna realizacija te raspodjele; ovdje se
    izvjestava ocekivanje, std i 95% raspon po izvlacenjima, kao i analiticko
    ocekivanje p_train*p_test + (1-p_train)*(1-p_test)."""
    rng = np.random.default_rng(seed)
    y_train, y_test = np.asarray(y_train), np.asarray(y_test)
    p_tr, p_te = float(y_train.mean()), float(y_test.mean())
    preds = rng.random((n_draws, len(y_test))) < p_tr
    accs = (preds == y_test.astype(bool)[None, :]).mean(axis=1)
    return {
        "p_class1_train": p_tr, "p_class1_test": p_te,
        "expected_accuracy_analytic": p_tr * p_te + (1 - p_tr) * (1 - p_te),
        "mean_accuracy": float(accs.mean()), "std_accuracy": float(accs.std(ddof=1)),
        "p2_5": float(np.percentile(accs, 2.5)), "p97_5": float(np.percentile(accs, 97.5)),
        "n_draws": int(n_draws),
    }


def holm_adjust(p_values) -> np.ndarray:
    """Holm-Bonferroni korekcija za visestruko testiranje (kontrola FWER)."""
    from statsmodels.stats.multitest import multipletests
    p = np.asarray(p_values, dtype=float)
    if len(p) == 0:
        return p
    return multipletests(p, method="holm")[1]


def mcnemar_family_table(y_true, predictions: dict, pairs: list, family: str) -> pd.DataFrame:
    """McNemar test za listu parova (a, b) iz `predictions` = {ime: y_pred}.
    Vraca tabelu sa sirovim i Holm-korigovanim p-vrijednostima unutar porodice."""
    rows = []
    for a, b in pairs:
        r = mcnemar_test(y_true, predictions[a], predictions[b])
        rows.append({"family": family, "model_a": a, "model_b": b,
                     "n_01": r["n_01"], "n_10": r["n_10"],
                     "statistic": r["statistic"], "p_value": r["p_value"]})
    df = pd.DataFrame(rows)
    if len(df):
        df["p_holm"] = holm_adjust(df["p_value"].values)
    return df


# ---------------------------------------------------------------------
# Cuvanje rezultata (results/predictions.csv, results/metrics.csv)
# ---------------------------------------------------------------------

def save_predictions_csv(model_predictions: dict, path: str) -> None:
    """Cuva SVE (y_true, y_pred) parove u jedan dugacki CSV - jedini
    izvor iz kojeg se kasnije mogu ponovo izracunati sve tabele/grafici
    bez ponovnog treniranja (mentorska tacka 1 i 7)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = []
    for name, (y_true, y_pred) in model_predictions.items():
        for i, (yt, yp) in enumerate(zip(y_true, y_pred)):
            rows.append({"model": name, "index": i, "y_true": int(yt), "y_pred": int(yp)})
    pd.DataFrame(rows).to_csv(path, index=False)


def save_metrics_csv(results_table: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    results_table.to_csv(path)