"""
Orkestracija cijelog eksperimenta - jedina funkcija koju notebook poziva.

Jedan tok automatski proizvodi sve rezultate (tabele, matrice konfuzije,
testovi, slike). 23-godisnji i 2-godisnji eksperiment pozivaju ISTU funkciju
run_experiment(); razlikuju se samo u cfg['data'] datumima. Kraci period je
podskup duzeg (leži unutar njegovog test perioda), pa se tretira kao
dopunska provjera osjetljivosti, a ne kao nezavisan eksperiment.

Protokol koji ova funkcija primjenjuje (jedinstven za sve modele):
  * Izbor konfiguracije: iskljucivo na D_val (kriterijum (24)); D_test se ne koristi.
  * Konacni model: fit na D_trainval = D_train + D_val, evaluacija na D_test.
  * Generalizacioni jaz: Acc(D_trainval) - Acc(D_test) ISTOG konacnog modela.
  * LSTM: n_stochastic_runs pokretanja; izvjestava se mean i std svih
    pokretanja. Za matricu konfuzije i testove koristi se pokretanje ciji je
    indeks zadat UNAPRIJED (lstm.representative_run_index, podrazumijevano 0),
    pa izbor ne zavisi od rezultata na testnom skupu.
"""
from __future__ import annotations
import os
import json
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from src.utils import set_global_seed, resolve_path
from src.data_loader import build_raw_dataset
from src.features import build_feature_matrix, dataset_summary, check_stationarity
from src.split import chronological_split, describe_cv_folds, make_xy
from src.models.sklearn_models import train_all_classical_models
from src.models.lstm_model import optimize_lstm, train_final_lstm_multi_run
from src.evaluation import (
    compute_all_baselines, compute_metrics, build_results_table, bootstrap_accuracy_ci,
    block_bootstrap_accuracy_ci, random_baseline_distribution, mcnemar_test,
    mcnemar_family_table, subperiod_accuracy, save_predictions_csv, save_metrics_csv,
)


def _json_safe(obj):
    """Zamjenjuje NaN/inf sa None (validan JSON) i numpy tipove sa python tipovima."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _lstm_runs_table(runs: list, label: str) -> pd.DataFrame:
    """Jedna tabela po konfiguraciji: metrike svakog pokretanja (isti model
    daje i in-sample tacnost na D_trainval i tacnost na D_test)."""
    rows = []
    for i, r in enumerate(runs):
        m = compute_metrics(r["y_true"], r["y_pred"])
        rows.append({
            "config": label, "run": i, "seed": r["seed"], "threshold": r["threshold_used"],
            "accuracy": m["accuracy"], "precision": m["precision"], "recall": m["recall"],
            "f1": m["f1"], "pred_rate_class1": r["pred_rate_class1"],
            "trainval_accuracy": r["trainval_accuracy"],
            "trainval_test_gap": r["trainval_accuracy"] - m["accuracy"],
        })
    return pd.DataFrame(rows)


def _mean_std(df: pd.DataFrame, cols: list) -> dict:
    return {c: {"mean": float(df[c].mean()),
                "std": float(df[c].std(ddof=1)) if len(df) > 1 else float("nan"),
                "min": float(df[c].min()), "max": float(df[c].max())} for c in cols}


def run_experiment(cfg: dict, fred_api_key: str, use_short_period: bool = False) -> dict:
    seed = cfg["random_seed"]
    set_global_seed(seed)

    data_cfg = dict(cfg["data"])
    if use_short_period:
        data_cfg["start_date"] = cfg["data"]["short_start_date"]
        data_cfg["end_date"] = cfg["data"]["short_end_date"]
        tag = "short_2y"
    else:
        tag = cfg["data"]["experiment_tag"]

    raw_path = resolve_path(cfg["paths"]["data_raw"], f"raw_{tag}.csv")
    raw = build_raw_dataset(fred_api_key, {**cfg, "data": data_cfg}, save_path=raw_path)

    processed_path = resolve_path(cfg["paths"]["data_processed"], f"model_df_{tag}.csv")
    model_df, feature_cols = build_feature_matrix(raw, save_path=processed_path)
    summary = dataset_summary(model_df, feature_cols)
    stationarity_path = resolve_path(cfg["paths"]["results"], f"stationarity_{tag}.csv")
    stationarity_table = check_stationarity(model_df, feature_cols, save_path=stationarity_path)

    gap = cfg["split"].get("gap", 1)
    train_df, val_df, test_df, split_info = chronological_split(
        model_df, cfg["split"]["train_fraction"], cfg["split"]["val_fraction"], gap=gap
    )
    fold_info = describe_cv_folds(train_df, cfg["split"]["n_cv_folds"], gap=gap)

    X_train, y_train = make_xy(train_df, feature_cols)
    X_val, y_val = make_xy(val_df, feature_cols)
    X_test, y_test = make_xy(test_df, feature_cols)
    X_trainval = np.concatenate([X_train, X_val])
    y_trainval = np.concatenate([y_train, y_val])

    return_idx = feature_cols.index("Return")
    trainval_df = pd.concat([train_df, val_df])
    returns_trainval = trainval_df["Return"].values
    returns_trainval_next = model_df["Return"].shift(-1).loc[trainval_df.index].values
    baseline_preds = compute_all_baselines(
        y_trainval, y_test, X_test[:, return_idx], seed,
        returns_trainval=returns_trainval, returns_trainval_next=returns_trainval_next,
    )
    random_dist = random_baseline_distribution(y_trainval, y_test, seed=seed)

    # ------------------------------------------------------------------
    # Klasicni modeli
    # ------------------------------------------------------------------
    classical_names = list(cfg["classical_models"]["param_grids"])
    classical_results = train_all_classical_models(
        X_train, y_train, X_val, y_val, X_trainval, y_trainval, X_test, y_test,
        cfg["classical_models"]["param_grids"], cfg["split"]["n_cv_folds"],
        cfg["classical_models"]["n_jobs"], seed, gap=gap,
        early_stopping_cfg=cfg["classical_models"].get("early_stopping"),
    )

    # ------------------------------------------------------------------
    # LSTM: izbor konfiguracije na D_val; zatim n_runs pokretanja KONACNOG
    # modela za izabranu i za fabricku konfiguraciju (isti broj pokretanja,
    # isti seedovi - da bi poredjenje prije/poslije bilo simetricno).
    # ------------------------------------------------------------------
    lstm_cfg = cfg["lstm"]
    default_lstm_params = lstm_cfg["default_hyperparameters"]
    n_runs = lstm_cfg["n_stochastic_runs"]
    rep_idx = int(lstm_cfg.get("representative_run_index", 0))   # pravilo zadato unaprijed
    if not 0 <= rep_idx < n_runs:
        raise ValueError("lstm.representative_run_index mora biti u [0, n_stochastic_runs).")

    best_lstm_params, best_lstm_val_acc, best_lstm_sel_train_acc, lstm_search_log = optimize_lstm(
        X_train, y_train, X_val, y_val, lstm_cfg["param_grid"],
        lstm_cfg["max_search_combinations"], lstm_cfg["epochs_search"], seed,
        default_params=default_lstm_params,
    )
    lstm_runs = train_final_lstm_multi_run(
        X_trainval, y_trainval, X_test, y_test, best_lstm_params,
        n_runs, lstm_cfg["epochs_final"], base_seed=seed,
    )
    lstm_default_runs = train_final_lstm_multi_run(
        X_trainval, y_trainval, X_test, y_test, default_lstm_params,
        n_runs, lstm_cfg["epochs_final"], base_seed=seed,
    )
    for r in lstm_runs + lstm_default_runs:
        if not np.array_equal(r["y_true"], y_test):
            raise ValueError("LSTM test ciljevi nisu identicni y_test - provjeriti poravnanje sekvenci.")
    rep_run = lstm_runs[rep_idx]
    rep_default_run = lstm_default_runs[rep_idx]

    lstm_selected_table = _lstm_runs_table(lstm_runs, "selected")
    lstm_default_table = _lstm_runs_table(lstm_default_runs, "default")
    lstm_runs_table = pd.concat([lstm_selected_table, lstm_default_table], ignore_index=True)
    run_cols = ["accuracy", "precision", "recall", "f1", "trainval_accuracy", "trainval_test_gap"]
    lstm_runs_summary = {
        "selected": _mean_std(lstm_selected_table, run_cols),
        "default": _mean_std(lstm_default_table, run_cols),
        "n_runs": n_runs,
        "representative_run_index": rep_idx,
        "representative_run_seed": rep_run["seed"],
        "representative_rule": ("pokretanje sa unaprijed zadatim indeksom (podrazumijevano prvo, "
                                "seed = random_seed); izbor ne koristi testni skup"),
    }

    # ------------------------------------------------------------------
    # Predikcije za tabele, matrice konfuzije i testove
    # ------------------------------------------------------------------
    all_predictions = {
        **{name: (y_test, classical_results["optimized_predictions"][name]) for name in classical_names},
        "lstm": (rep_run["y_true"], rep_run["y_pred"]),
        **{name: (y_test, pred) for name, pred in baseline_preds.items()},
    }
    results_table = build_results_table(all_predictions)
    model_names = classical_names + ["lstm"]
    baseline_names = list(baseline_preds.keys())

    # ---- Prije / poslije optimizacije ---------------------------------
    optimization_rows = []
    for name in classical_names:
        before_acc = float((classical_results["baseline_predictions"][name] == y_test).mean())
        after_acc = float(results_table.loc[name, "accuracy"])
        optimization_rows.append({
            "model": name, "accuracy_before_optimization": round(before_acc, 4),
            "accuracy_after_optimization": round(after_acc, 4),
            "improved": bool(after_acc > before_acc),
        })
    # LSTM: srednja tacnost svih pokretanja (isti broj pokretanja za obje konfiguracije)
    lstm_before_acc = float(lstm_default_table["accuracy"].mean())
    lstm_after_acc = float(lstm_selected_table["accuracy"].mean())
    optimization_rows.append({
        "model": "lstm", "accuracy_before_optimization": round(lstm_before_acc, 4),
        "accuracy_after_optimization": round(lstm_after_acc, 4),
        "improved": bool(lstm_after_acc > lstm_before_acc),
        "std_before": round(float(lstm_default_table["accuracy"].std(ddof=1)), 4),
        "std_after": round(float(lstm_selected_table["accuracy"].std(ddof=1)), 4),
    })
    optimization_comparison = pd.DataFrame(optimization_rows).set_index("model")
    n_improved = int(optimization_comparison["improved"].sum())
    n_total = len(optimization_comparison)
    best_model = results_table.loc[model_names, "accuracy"].idxmax()   # samo za sliku/opis

    # ---- Intervali pouzdanosti ---------------------------------------
    block_len = int(cfg.get("statistics", {}).get("block_length", 10))
    stats = {}
    ci_rows = []
    for name in model_names + baseline_names:
        yt, yp = all_predictions[name]
        iid = bootstrap_accuracy_ci(yt, yp, seed=seed)
        blk = block_bootstrap_accuracy_ci(yt, yp, block_length=block_len, seed=seed)
        stats[f"bootstrap_ci_{name}"] = iid
        stats[f"blockboot_ci_{name}"] = blk
        ci_rows.append({"model": name, "accuracy": iid["point_estimate"],
                        "ci_lower": iid["ci_lower"], "ci_upper": iid["ci_upper"],
                        "ci_block_lower": blk["ci_lower"], "ci_block_upper": blk["ci_upper"],
                        "block_length": blk["block_length"]})
    ci_table = pd.DataFrame(ci_rows).set_index("model")

    # ---- McNemar: svi modeli vs. svi bazni; svi parovi modela -----------
    pred_only = {n: all_predictions[n][1] for n in model_names + baseline_names}
    fam_base = mcnemar_family_table(
        y_test, pred_only, [(m, b) for m in model_names for b in baseline_names], "model_vs_baseline")
    fam_pair = mcnemar_family_table(
        y_test, pred_only, list(itertools.combinations(model_names, 2)), "model_vs_model")
    mcnemar_table = pd.concat([fam_base, fam_pair], ignore_index=True)
    for _, row in mcnemar_table.iterrows():
        stats[f"mcnemar_{row['model_a']}_vs_{row['model_b']}"] = {
            "statistic": float(row["statistic"]), "p_value": float(row["p_value"]),
            "p_holm": float(row["p_holm"]), "n_01": int(row["n_01"]), "n_10": int(row["n_10"]),
        }

    # Osjetljivost na izbor LSTM pokretanja: McNemar za SVAKO od n_runs pokretanja
    lstm_run_tests = []
    for i, r in enumerate(lstm_runs):
        for opp in baseline_names + classical_names:
            res = mcnemar_test(y_test, r["y_pred"], pred_only[opp])
            lstm_run_tests.append({"run": i, "seed": r["seed"], "vs": opp, **res})
    lstm_run_tests_table = pd.DataFrame(lstm_run_tests)

    # ---- Zbirna tabela svih modela ------------------------------------
    summary_rows = []
    for name in model_names + baseline_names:
        m = results_table.loc[name]
        row = {"model": name, "kind": "baseline" if name.startswith("baseline_") else "model",
               "n": int(m["n_samples"]), "accuracy": float(m["accuracy"]),
               "precision": float(m["precision"]), "recall": float(m["recall"]), "f1": float(m["f1"]),
               "ci_lower": float(ci_table.loc[name, "ci_lower"]),
               "ci_upper": float(ci_table.loc[name, "ci_upper"]),
               "ci_block_lower": float(ci_table.loc[name, "ci_block_lower"]),
               "ci_block_upper": float(ci_table.loc[name, "ci_block_upper"])}
        for b in baseline_names:
            hit = fam_base[(fam_base["model_a"] == name) & (fam_base["model_b"] == b)]
            row[f"p_mcnemar_vs_{b}"] = float(hit["p_value"].iloc[0]) if len(hit) else np.nan
        if name == "lstm":
            row["accuracy_mean_runs"] = lstm_runs_summary["selected"]["accuracy"]["mean"]
            row["accuracy_std_runs"] = lstm_runs_summary["selected"]["accuracy"]["std"]
        summary_rows.append(row)
    summary_table = pd.DataFrame(summary_rows).set_index("model")

    # ---- Generalizacioni jaz: ISTI konacni model (fit na D_trainval) -----
    generalization_rows = []
    for name in classical_names:
        trainval_acc = accuracy_score(y_trainval, classical_results["trainval_predictions"][name])
        test_acc = float(results_table.loc[name, "accuracy"])
        sel_train = accuracy_score(y_train, classical_results["selection_train_predictions"][name])
        sel_val = accuracy_score(y_val, classical_results["selection_val_predictions"][name])
        generalization_rows.append({
            "model": name,
            "trainval_accuracy": round(trainval_acc, 4), "test_accuracy": round(test_acc, 4),
            "trainval_test_gap": round(trainval_acc - test_acc, 4),
            "selection_train_accuracy": round(sel_train, 4), "selection_val_accuracy": round(sel_val, 4),
        })
    lstm_test_acc = float(results_table.loc["lstm", "accuracy"])
    generalization_rows.append({
        "model": "lstm",
        "trainval_accuracy": round(rep_run["trainval_accuracy"], 4), "test_accuracy": round(lstm_test_acc, 4),
        "trainval_test_gap": round(rep_run["trainval_accuracy"] - lstm_test_acc, 4),
        "selection_train_accuracy": round(best_lstm_sel_train_acc, 4),
        "selection_val_accuracy": round(best_lstm_val_acc, 4),
    })
    generalization_table = pd.DataFrame(generalization_rows).set_index("model")
    generalization_table = generalization_table.sort_values("trainval_test_gap", ascending=False)

    # ---- Tacnost po potperiodima testnog skupa -------------------------
    n_subperiods = 3
    subperiod_tables = {}
    for name in model_names + baseline_names:
        y_true_m, y_pred_m = all_predictions[name]
        subperiod_tables[name] = subperiod_accuracy(y_true_m, y_pred_m, test_df.index, n_subperiods)

    # ------------------------------------------------------------------
    # Cuvanje tabela
    # ------------------------------------------------------------------
    res_dir = lambda fname: resolve_path(cfg["paths"]["results"], fname)
    save_predictions_csv(all_predictions, res_dir(f"predictions_{tag}.csv"))
    save_metrics_csv(results_table, res_dir(f"metrics_{tag}.csv"))
    generalization_table.to_csv(res_dir(f"generalization_{tag}.csv"))
    optimization_comparison.to_csv(res_dir(f"optimization_comparison_{tag}.csv"))
    lstm_runs_table.to_csv(res_dir(f"lstm_runs_{tag}.csv"), index=False)
    lstm_run_tests_table.to_csv(res_dir(f"lstm_runs_mcnemar_{tag}.csv"), index=False)
    mcnemar_table.to_csv(res_dir(f"mcnemar_{tag}.csv"), index=False)
    ci_table.to_csv(res_dir(f"confidence_intervals_{tag}.csv"))
    summary_table.to_csv(res_dir(f"summary_table_{tag}.csv"))

    subperiod_long = []
    for name, tbl in subperiod_tables.items():
        tbl = tbl.copy()
        tbl.insert(0, "model", name)
        subperiod_long.append(tbl)
    pd.concat(subperiod_long, ignore_index=True).to_csv(res_dir(f"subperiod_accuracy_{tag}.csv"), index=False)

    # ------------------------------------------------------------------
    # Slike
    # ------------------------------------------------------------------
    figures_dir = os.path.dirname(resolve_path(cfg["paths"]["figures"], "placeholder.png"))
    _save_figures(results_table, all_predictions, best_model, baseline_preds, figures_dir, tag)
    _save_all_confusion_matrices(all_predictions, model_names, figures_dir, tag)

    before_after_predictions = {
        name: (y_test, classical_results["baseline_predictions"][name],
               classical_results["optimized_predictions"][name])
        for name in classical_names
    }
    # LSTM: isto (unaprijed zadato) pokretanje za obje konfiguracije
    before_after_predictions["lstm"] = (y_test, rep_default_run["y_pred"], rep_run["y_pred"])
    _save_before_after_confusion_matrices(before_after_predictions, figures_dir, tag)
    _save_individual_before_after_matrices(before_after_predictions, figures_dir, tag)

    out = {
        "tag": tag,
        "protocol": {
            "prediction_moment": "nakon zatvaranja na dan t; Close(t) poznat, cilj: smjer Close(t+1)",
            "fred_series_lag_days": cfg["fred"]["series_lag_days"],
            "fred_vintage_series": cfg["fred"]["vintage_series"],
            "lstm_n_runs": n_runs,
            "lstm_representative_run_index": rep_idx,
            "bootstrap_block_length": block_len,
            "generalization_gap_definition": "Acc(D_trainval) - Acc(D_test), isti konacni model",
        },
        "dataset_summary": summary,
        "split_info": split_info,
        "cv_fold_info": fold_info,
        "best_hyperparameters": {**classical_results["best_params"], "lstm": best_lstm_params},
        "lstm_search_log": lstm_search_log,
        "early_stopping_info": classical_results["early_stopping_info"],
        "model_selection": classical_results["model_selection"],
        "final_params_used": classical_results["final_params_used"],
        "lstm_n_test_sequences": rep_run["n_test_sequences"],
        "lstm_runs": {"per_run": lstm_runs_table.to_dict(orient="records"), "summary": lstm_runs_summary},
        "results_table": results_table.reset_index().to_dict(orient="records"),
        "summary_table": summary_table.reset_index().to_dict(orient="records"),
        "confidence_intervals": ci_table.reset_index().to_dict(orient="records"),
        "statistical_tests": stats,
        "mcnemar_table": mcnemar_table.to_dict(orient="records"),
        "lstm_runs_mcnemar": lstm_run_tests_table.to_dict(orient="records"),
        "random_baseline_distribution": random_dist,
        "generalization_table": generalization_table.reset_index().to_dict(orient="records"),
        "optimization_comparison": optimization_comparison.reset_index().to_dict(orient="records"),
        "n_models_improved": f"{n_improved}/{n_total}",
        "subperiod_accuracy": {name: tbl.to_dict(orient="records") for name, tbl in subperiod_tables.items()},
        "stationarity_table": stationarity_table.reset_index().to_dict(orient="records"),
    }

    with open(res_dir(f"summary_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(_json_safe(out), f, indent=2, default=str)

    return out


def _save_figures(results_table, all_predictions, best_model, baseline_preds, figures_dir, tag):
    os.makedirs(figures_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    results_table["accuracy"].plot(kind="bar", ax=axes[0], color="steelblue")
    majority_acc = (baseline_preds["baseline_majority"] ==
                     all_predictions["baseline_majority"][0]).mean()
    axes[0].axhline(majority_acc, color="red", linestyle="--", label="baseline (majority)")
    axes[0].set_ylabel("Tacnost")
    axes[0].set_title("Poredjenje tacnosti svih modela")
    axes[0].legend()
    axes[0].tick_params(axis="x", rotation=45)
    cm = confusion_matrix(*all_predictions[best_model], labels=[0, 1])
    axes[1].imshow(cm, cmap="Blues")
    axes[1].set_title(f"Matrica konfuzije - {best_model}")
    axes[1].set_xticks([0, 1]); axes[1].set_xticklabels(["Pad", "Rast"])
    axes[1].set_yticks([0, 1]); axes[1].set_yticklabels(["Pad", "Rast"])
    for i in range(2):
        for j in range(2):
            axes[1].text(j, i, cm[i, j], ha="center", va="center", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, f"results_comparison_{tag}.png"), dpi=150)
    plt.close(fig)


def _plot_cm_on_axis(ax, y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    ax.imshow(cm, cmap="Blues")
    ax.set_title(title, fontsize=10)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Pad", "Rast"], fontsize=8)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Pad", "Rast"], fontsize=8)
    for r in range(2):
        for c in range(2):
            ax.text(c, r, int(cm[r, c]), ha="center", va="center", fontsize=11)


def _save_all_confusion_matrices(all_predictions, model_names, figures_dir, tag):
    os.makedirs(figures_dir, exist_ok=True)
    n = len(model_names)
    fig, axes = plt.subplots(n, 1, figsize=(5, 4 * n))
    if n == 1:
        axes = [axes]
    for ax, name in zip(axes, model_names):
        y_true, y_pred = all_predictions[name]
        _plot_cm_on_axis(ax, y_true, y_pred, f"Matrica konfuzije - {name}")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, f"all_confusion_matrices_{tag}.png"), dpi=150)
    plt.close(fig)


def _save_before_after_confusion_matrices(before_after_predictions, figures_dir, tag):
    os.makedirs(figures_dir, exist_ok=True)
    model_names = list(before_after_predictions.keys())
    n = len(model_names)
    fig, axes = plt.subplots(n, 2, figsize=(8, 4 * n))
    if n == 1:
        axes = axes.reshape(1, 2)
    for i, name in enumerate(model_names):
        y_true, y_before, y_after = before_after_predictions[name]
        _plot_cm_on_axis(axes[i, 0], y_true, y_before, f"{name} - PRIJE optimizacije")
        _plot_cm_on_axis(axes[i, 1], y_true, y_after, f"{name} - POSLIJE optimizacije")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, f"optimization_before_after_{tag}.png"), dpi=150)
    plt.close(fig)


def _save_individual_before_after_matrices(before_after_predictions, figures_dir, tag):
    os.makedirs(figures_dir, exist_ok=True)
    for name, (y_true, y_before, y_after) in before_after_predictions.items():
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        _plot_cm_on_axis(axes[0], y_true, y_before, f"{name} - PRIJE optimizacije")
        _plot_cm_on_axis(axes[1], y_true, y_after, f"{name} - POSLIJE optimizacije")
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, f"optimization_before_after_{name}_{tag}.png"), dpi=150)
        plt.close(fig)