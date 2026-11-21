"""
figures.py - slike za Poglavlje 5 (Slike 4-9) iz sacuvanih rezultata eksperimenta.

Ne trenira nista: cita results/summary_full_23y.json i results/summary_short_2y.json i crta ono sto je eksperiment sacuvao.
Isti kod (bez izmjena u crtanju) kao ranije samostalna skripta make_figures_v2.py, kojom su napravljene slike u radu.

Iz notebook-a:

    from src.figures import make_all_figures, load
    make_all_figures(load("results/summary_full_23y.json"), load("results/summary_short_2y.json"), "results/figures_rad")

Iz komandne linije (iz korijena projekta):

    python -m src.figures
    python -m src.figures results/summary_full_23y.json results/summary_short_2y.json results/figures_rad

Izlaz (sest slika, redom kao u radu):
    slika4_optimizacija.png              Slika 4  tacnost prije i poslije optimizacije
    slika5_rangirana_tacnost.png         Slika 5  rangirana tacnost svih devet modela
    slika6_matrice_konfuzije_puni.png    Slika 6  matrice konfuzije, puni eksperiment
    slika7_matrice_konfuzije_kraci.png   Slika 7  matrice konfuzije, kraci eksperiment
    slika8_generalizacioni_jaz.png       Slika 8  generalizacioni jaz (trening+val. naspram testa)
    slika9_potperiodi.png                Slika 9  tacnost kroz tri potperioda
"""
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

BLUE, GRAY, RED = "#2c5f8a", "#8c8c8c", "#c0392b"
NAMES = {"logreg": "Logistička regresija", "svm": "SVM", "xgboost": "XGBoost", "lightgbm": "LightGBM", "lstm": "LSTM",
         "baseline_persistence": "Perzistencija*", "baseline_majority": "Većinska klasa*",
         "baseline_random": "Slučajni klasifikator*", "baseline_ar1": "AR(1)*"}
SHORT = dict(NAMES, baseline_majority="Većinska*", baseline_random="Slučajni*", logreg="Logistička\nregresija",
             lstm="LSTM\n(pokretanje 0)")
MODELS = ["logreg", "svm", "xgboost", "lightgbm", "lstm"]
ALL = MODELS + ["baseline_persistence", "baseline_majority", "baseline_random", "baseline_ar1"]
TITLES = ("Puni eksperiment", "Kraći eksperiment")
BASELINE_NOTE = r"Referentni ($\mathit{baseline}$) model *"


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------------ Slika 4: prije/poslije optimizacije
def fig_optimization(full, short, out):
    labels = [NAMES[m] if m != "lstm" else "LSTM\n(srednja vr.)" for m in MODELS]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    x = np.arange(len(labels))
    width = 0.35
    for ax, res, title in zip(axes, (full, short), TITLES):
        comp = {r["model"]: r for r in res["optimization_comparison"]}
        before = [comp[m]["accuracy_before_optimization"] for m in MODELS]
        after = [comp[m]["accuracy_after_optimization"] for m in MODELS]
        b1 = ax.bar(x - width / 2, before, width, label="Prije optimizacije", color=GRAY)
        b2 = ax.bar(x + width / 2, after, width, label="Poslije optimizacije", color=BLUE)
        ax.axhline(0.5, color="black", linewidth=0.8, linestyle=":", alpha=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(title)
        ax.set_ylabel("Tačnost")
        ax.grid(axis="y", alpha=0.25)
        for bar in list(b1) + list(b2):
            h = bar.get_height()
            ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=8)
    fig.suptitle("Tačnost modela prije i poslije optimizacije hiperparametara", fontsize=13)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower right", bbox_to_anchor=(0.98, 0.02), fontsize=9, frameon=True)
    plt.tight_layout(rect=[0, 0.10, 1, 0.93])
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ Slika 5: rangirana tačnost
def fig_ranked(full, short, out):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, res, title in zip(axes, (full, short), TITLES):
        data = sorted([(r["model"], r["accuracy"]) for r in res["results_table"]], key=lambda t: t[1])
        names = [NAMES[m] for m, _ in data]
        vals = [v for _, v in data]
        colors = [GRAY if m.startswith("baseline_") else BLUE for m, _ in data]
        y = np.arange(len(names))
        bars = ax.barh(y, vals, color=colors)
        ax.set_yticks(y)
        ax.set_yticklabels(names)
        ax.set_xlim(0.40, 0.63)
        ax.axvline(0.5, color="black", linewidth=0.8, linestyle=":", alpha=0.6)
        ax.set_title(title)
        ax.set_xlabel("Tačnost")
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.4f}", xy=(v, b.get_y() + b.get_height() / 2), xytext=(4, 0),
                        textcoords="offset points", va="center", fontsize=8)
    legend_elems = [Patch(facecolor=BLUE, label="Testirani model"), Patch(facecolor=GRAY, label=BASELINE_NOTE)]
    fig.legend(handles=legend_elems, loc="lower center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Rangirana tačnost svih modela na testnom skupu", fontsize=13)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ Slike 6 i 7: matrice konfuzije
def fig_confusion(res, title, out):
    data = {r["model"]: (r["TN"], r["FP"], r["FN"], r["TP"]) for r in res["results_table"]}
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    for ax, m in zip(axes.flat, ALL):
        tn, fp, fn, tp = data[m]
        mat = np.array([[tn, fp], [fn, tp]])
        total = mat.sum()
        ax.imshow(mat, cmap="Blues", vmin=0)
        for i in range(2):
            for j in range(2):
                val = mat[i, j]
                color = "white" if val > mat.max() * 0.5 else "black"
                ax.text(j, i, f"{val}\n({val / total * 100:.0f}%)", ha="center", va="center", color=color, fontsize=9)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["pad", "rast"], fontsize=8)
        ax.set_yticks([0, 1]); ax.set_yticklabels(["pad", "rast"], fontsize=8)
        ax.set_title(SHORT[m], fontsize=10)
    fig.suptitle(title, fontsize=13)
    fig.text(0.5, 0.01, r"redovi = stvarno, kolone = predviđeno   |   * = referentni ($\mathit{baseline}$) model",
             ha="center", fontsize=8)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ Slika 8: generalizacioni jaz
def fig_gap(full, short, out):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, res, title in zip(axes, (full, short), TITLES):
        data = {r["model"]: (r["trainval_accuracy"], r["test_accuracy"], r["trainval_test_gap"])
                for r in res["generalization_table"]}
        order = sorted(data, key=lambda m: data[m][2])          # po jazu, rastuće
        x = np.arange(len(order))
        w = 0.35
        ax.bar(x - w / 2, [data[m][0] for m in order], w, label="Tačnost na trening+val. skupu", color=RED, alpha=0.85)
        ax.bar(x + w / 2, [data[m][1] for m in order], w, label="Tačnost na testnom skupu", color=BLUE)
        for xi, m in zip(x, order):
            ax.annotate(f"jaz = {data[m][2]:.3f}", xy=(xi, max(data[m][0], data[m][1]) + 0.02), ha="center",
                        fontsize=8, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([("LSTM\n(pokretanje 0)" if m == "lstm" else NAMES[m]) for m in order], fontsize=8)
        ax.set_ylim(0, 1.1)
        ax.set_title(title)
        ax.set_ylabel("Tačnost")
        ax.axhline(0.5, color="black", linewidth=0.7, linestyle=":", alpha=0.5)
    axes[0].legend(loc="upper left", fontsize=9)
    fig.suptitle("Generalizacioni jaz: tačnost na trening+val. i na testnom skupu", fontsize=13)
    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ Slika 9: potperiodi
def fig_subperiod(full, short, out):
    colors = {"logreg": "#2c5f8a", "svm": "#c0392b", "xgboost": "#27ae60", "lightgbm": "#8e44ad", "lstm": "#e67e22"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    for ax, res, title in zip(axes, (full, short), TITLES):
        sub = res["subperiod_accuracy"]
        for m in MODELS:
            vals = [p["accuracy"] for p in sorted(sub[m], key=lambda p: p["period"])]
            ax.plot([1, 2, 3], vals, marker="o", label=("LSTM (pokretanje 0)" if m == "lstm" else NAMES[m]),
                    color=colors[m], linewidth=2)
        ax.axhline(0.5, color="black", linewidth=0.7, linestyle=":", alpha=0.5)
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(["Potperiod 1", "Potperiod 2", "Potperiod 3"])
        ax.set_title(title)
        ax.set_ylabel("Tačnost")
    axes[0].legend(loc="best", fontsize=8, ncol=2)
    fig.suptitle("Tačnost modela kroz tri uzastopna vremenska potperioda", fontsize=13)
    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ sve slike odjednom
def make_all_figures(full, short, out_dir):
    """Crta Slike 4-9 iz rjecnika `full` i `short` (sadrzaj summary_*.json) u folder `out_dir`. Vraca listu putanja."""
    os.makedirs(out_dir, exist_ok=True)
    jobs = [
        ("slika4_optimizacija.png", lambda p: fig_optimization(full, short, p)),
        ("slika5_rangirana_tacnost.png", lambda p: fig_ranked(full, short, p)),
        ("slika6_matrice_konfuzije_puni.png", lambda p: fig_confusion(full, "Matrice konfuzije, puni eksperiment", p)),
        ("slika7_matrice_konfuzije_kraci.png", lambda p: fig_confusion(short, "Matrice konfuzije, kraći eksperiment", p)),
        ("slika8_generalizacioni_jaz.png", lambda p: fig_gap(full, short, p)),
        ("slika9_potperiodi.png", lambda p: fig_subperiod(full, short, p)),
    ]
    paths = []
    for name, draw in jobs:
        path = os.path.join(out_dir, name)
        draw(path)
        paths.append(path)
    return paths


if __name__ == "__main__":
    plt.switch_backend("Agg")
    args = sys.argv[1:] or ["results/summary_full_23y.json", "results/summary_short_2y.json", "results/figures_rad"]
    written = make_all_figures(load(args[0]), load(args[1]), args[2])
    print("OK ->", args[2], f"({len(written)} slika)")
