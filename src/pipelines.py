from __future__ import annotations
import os
import json
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
    compute_all_baselines, build_results_table, bootstrap_accuracy_ci,
    mcnemar_test, subperiod_accuracy, save_predictions_csv, save_metrics_csv,
)


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

    classical_results = train_all_classical_models(
        X_train, y_train, X_val, y_val, X_trainval, y_trainval, X_test, y_test,
        cfg["classical_models"]["param_grids"], cfg["split"]["n_cv_folds"],
        cfg["classical_models"]["n_jobs"], seed, gap=gap,
        early_stopping_cfg=cfg["classical_models"].get("early_stopping"),
    )

    lstm_cfg = cfg["lstm"]
    default_lstm_params = cfg["lstm"]["default_hyperparameters"]
    best_lstm_params, best_lstm_val_acc, best_lstm_train_acc, lstm_search_log = optimize_lstm(
        X_train, y_train, X_val, y_val, lstm_cfg["param_grid"],
        lstm_cfg["max_search_combinations"], lstm_cfg["epochs_search"], seed,
        default_params=default_lstm_params,
    )
    lstm_runs = train_final_lstm_multi_run(
        X_trainval, y_trainval, X_test, y_test, best_lstm_params,
        lstm_cfg["n_stochastic_runs"], lstm_cfg["epochs_final"],
        base_seed=seed,
    )
    lstm_accs = [r["accuracy"] for r in lstm_runs]
    median_run = min(lstm_runs, key=lambda r: abs(r["accuracy"] - np.mean(lstm_accs)))

    lstm_baseline_run = train_final_lstm_multi_run(
        X_trainval, y_trainval, X_test, y_test, default_lstm_params,
        n_runs=1, epochs=lstm_cfg["epochs_final"], base_seed=seed,
    )[0]

    all_predictions = {
        **{name: (y_test, classical_results["optimized_predictions"][name])
           for name in cfg["classical_models"]["param_grids"]},
        "lstm": (median_run["y_true"], median_run["y_pred"]),
        **{name: (y_test, pred) for name, pred in baseline_preds.items()},
    }

    results_table = build_results_table(all_predictions)

    optimization_rows = []
    for name in cfg["classical_models"]["param_grids"]:
        before_acc = float((classical_results["baseline_predictions"][name] == y_test).mean())
        after_acc = float(results_table.loc[name, "accuracy"])
        optimization_rows.append({
            "model": name, "accuracy_before_optimization": round(before_acc, 4),
            "accuracy_after_optimization": round(after_acc, 4),
            "improved": bool(after_acc > before_acc),
        })
    lstm_before_acc = float(lstm_baseline_run["accuracy"])
    lstm_after_acc = float(results_table.loc["lstm", "accuracy"])
    optimization_rows.append({
        "model": "lstm", "accuracy_before_optimization": round(lstm_before_acc, 4),
        "accuracy_after_optimization": round(lstm_after_acc, 4),
        "improved": bool(lstm_after_acc > lstm_before_acc),
    })
    optimization_comparison = pd.DataFrame(optimization_rows).set_index("model")
    n_improved = int(optimization_comparison["improved"].sum())
    n_total = len(optimization_comparison)
    non_baseline = [m for m in results_table.index if not m.startswith("baseline_")]
    best_model = results_table.loc[non_baseline, "accuracy"].idxmax()
    stats = {}
    for baseline_name in ["baseline_majority", "baseline_random", "baseline_persistence", "baseline_ar1"]:
        stats[f"mcnemar_{best_model}_vs_{baseline_name}"] = mcnemar_test(
            y_test, all_predictions[best_model][1], all_predictions[baseline_name][1]
        )
    for name in list(cfg["classical_models"]["param_grids"]) + ["lstm"]:
        stats[f"bootstrap_ci_{name}"] = bootstrap_accuracy_ci(*all_predictions[name], seed=seed)

    complex_models = ["xgboost", "lightgbm", "lstm"]
    linear_models = ["logreg", "svm"]
    for complex_name in complex_models:
        for linear_name in linear_models:
            stats[f"mcnemar_{complex_name}_vs_{linear_name}"] = mcnemar_test(
                y_test, all_predictions[complex_name][1], all_predictions[linear_name][1]
            )

    generalization_rows = []
    for name in cfg["classical_models"]["param_grids"]:
        train_acc = accuracy_score(y_train, classical_results["train_predictions"][name])
        val_acc = accuracy_score(y_val, classical_results["val_predictions"][name])
        test_acc = float(results_table.loc[name, "accuracy"])
        generalization_rows.append({
            "model": name, "train_accuracy": round(train_acc, 4),
            "val_accuracy": round(val_acc, 4), "test_accuracy": round(test_acc, 4),
            "train_test_gap": round(train_acc - test_acc, 4),
        })
    lstm_test_acc = float(results_table.loc["lstm", "accuracy"])
    generalization_rows.append({
        "model": "lstm", "train_accuracy": round(best_lstm_train_acc, 4),
        "val_accuracy": round(best_lstm_val_acc, 4), "test_accuracy": round(lstm_test_acc, 4),
        "train_test_gap": round(best_lstm_train_acc - lstm_test_acc, 4),
    })
    generalization_table = pd.DataFrame(generalization_rows).set_index("model")
    generalization_table = generalization_table.sort_values("train_test_gap", ascending=False)

    n_subperiods = 3
    subperiod_tables = {}
    for name in list(cfg["classical_models"]["param_grids"]) + ["lstm"] + list(baseline_preds.keys()):
        y_true_m, y_pred_m = all_predictions[name]
        subperiod_tables[name] = subperiod_accuracy(y_true_m, y_pred_m, test_df.index, n_subperiods)

    save_predictions_csv(all_predictions, resolve_path(cfg["paths"]["results"], f"predictions_{tag}.csv"))
    save_metrics_csv(results_table, resolve_path(cfg["paths"]["results"], f"metrics_{tag}.csv"))

    generalization_path = resolve_path(cfg["paths"]["results"], f"generalization_{tag}.csv")
    generalization_table.to_csv(generalization_path)

    optimization_comparison_path = resolve_path(cfg["paths"]["results"], f"optimization_comparison_{tag}.csv")
    optimization_comparison.to_csv(optimization_comparison_path)

    subperiod_path = resolve_path(cfg["paths"]["results"], f"subperiod_accuracy_{tag}.csv")
    subperiod_long = []
    for name, tbl in subperiod_tables.items():
        tbl = tbl.copy()
        tbl.insert(0, "model", name)
        subperiod_long.append(tbl)
    pd.concat(subperiod_long, ignore_index=True).to_csv(subperiod_path, index=False)

    figures_dir = os.path.dirname(resolve_path(cfg["paths"]["figures"], "placeholder.png"))
    _save_figures(results_table, all_predictions, best_model, baseline_preds, figures_dir, tag)

    main_models = list(cfg["classical_models"]["param_grids"]) + ["lstm"]
    _save_all_confusion_matrices(all_predictions, main_models, figures_dir, tag)

    before_after_predictions = {
        name: (y_test, classical_results["baseline_predictions"][name],
               classical_results["optimized_predictions"][name])
        for name in cfg["classical_models"]["param_grids"]
    }
    before_after_predictions["lstm"] = (
        y_test, lstm_baseline_run["y_pred"], median_run["y_pred"]
    )
    _save_before_after_confusion_matrices(before_after_predictions, figures_dir, tag)
    _save_individual_before_after_matrices(before_after_predictions, figures_dir, tag)

    out = {
        "tag": tag,
        "dataset_summary": summary,
        "split_info": split_info,
        "cv_fold_info": fold_info,
        "best_hyperparameters": {**classical_results["best_params"], "lstm": best_lstm_params},
        "lstm_search_log": lstm_search_log,
        "early_stopping_info": classical_results["early_stopping_info"],
        "model_selection": classical_results["model_selection"],
        "final_params_used": classical_results["final_params_used"],
        "lstm_n_test_sequences": median_run["n_test_sequences"],
        "results_table": results_table.reset_index().to_dict(orient="records"),
        "statistical_tests": stats,
        "generalization_table": generalization_table.reset_index().to_dict(orient="records"),
        "optimization_comparison": optimization_comparison.reset_index().to_dict(orient="records"),
        "n_models_improved": f"{n_improved}/{n_total}",
        "subperiod_accuracy": {name: tbl.to_dict(orient="records") for name, tbl in subperiod_tables.items()},
        "stationarity_table": stationarity_table.reset_index().to_dict(orient="records"),
    }

    summary_path = resolve_path(cfg["paths"]["results"], f"summary_{tag}.json")
    with open(summary_path, "w") as f:
        json.dump(out, f, indent=2, default=str)

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