from __future__ import annotations
import itertools
import os
import tempfile
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.utils.class_weight import compute_class_weight
from joblib import Parallel, delayed

def make_sequences(X: np.ndarray, y: np.ndarray, seq_len: int):
    n = len(X) - seq_len + 1
    if n <= 0:
        raise ValueError("Duzina sekvence veca je od raspolozivog broja uzoraka.")
    if len(X) != len(y):
        raise ValueError("X i y moraju imati isti broj uzoraka da bi sekvence bile usaglasene.")
    X_seq = np.stack([X[i:i + seq_len] for i in range(n)])
    y_seq = y[seq_len - 1: seq_len - 1 + n]
    return X_seq, y_seq

def get_class_weight_dict(y: np.ndarray) -> dict:
    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    return {int(c): float(w) for c, w in zip(classes, weights)}


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray,
                         thresholds: np.ndarray | None = None) -> tuple[float, float]:
    if thresholds is None:
        thresholds = np.arange(0.30, 0.71, 0.02)
    best_thr, best_acc = 0.5, -1.0
    for thr in thresholds:
        preds = (y_prob >= thr).astype(int)
        acc = (preds == y_true).mean()
        if acc > best_acc:
            best_acc, best_thr = float(acc), float(thr)
    return best_thr, best_acc

def build_lstm(input_shape, n_units: int, dropout: float, learning_rate: float, seed: int,
                n_layers: int = 1):
    """Gradi LSTM mrezu sa n_layers slojeva (mentorska tacka 6: broj
    slojeva mora biti dio pretrage hiperparametara, ne fiksiran)."""
    tf.random.set_seed(seed)
    lstm_layers = []
    for i in range(n_layers):
        is_last = (i == n_layers - 1)
        lstm_layers.append(layers.LSTM(n_units, dropout=dropout, return_sequences=not is_last))

    model = models.Sequential([
        layers.Input(shape=input_shape),
        *lstm_layers,
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model

def _extend_for_sequence(X_train, y_train, X_other, y_other, seq_len):
    if seq_len <= 1:
        return X_other, y_other
    X_ext = np.concatenate([X_train[-(seq_len - 1):], X_other])
    y_ext = np.concatenate([y_train[-(seq_len - 1):], y_other])
    return X_ext, y_ext

def _fit_with_checkpoint(model, X_fit, y_fit, validation_data=None, *, epochs, batch_size,
                          class_weight, monitor: str):
 
    es = callbacks.EarlyStopping(monitor=monitor, patience=5, restore_best_weights=True)
    rlrop = callbacks.ReduceLROnPlateau(monitor=monitor, factor=0.5, patience=3, min_lr=1e-5)

    fd, checkpoint_path = tempfile.mkstemp(suffix=".weights.h5")
    os.close(fd)
    mc = callbacks.ModelCheckpoint(
        filepath=checkpoint_path, monitor=monitor,
        save_best_only=True, save_weights_only=True,
    )

    fit_kwargs = dict(epochs=epochs, batch_size=batch_size,
                       callbacks=[es, rlrop, mc], class_weight=class_weight, verbose=0)
    if validation_data is not None:
        fit_kwargs["validation_data"] = validation_data

    model.fit(X_fit, y_fit, **fit_kwargs)

    if os.path.exists(checkpoint_path):
        model.load_weights(checkpoint_path)  
        os.remove(checkpoint_path)

    return model


def _evaluate_lstm_candidate(X_train, y_train, X_val, y_val, params: dict,
                              epochs: int, seed: int, use_class_weight: bool):
    
    seq_len = params["sequence_length"]

    Xtr_seq, ytr_seq = make_sequences(X_train, y_train, seq_len)
    X_val_ext, y_val_ext = _extend_for_sequence(X_train, y_train, X_val, y_val, seq_len)
    Xval_seq, yval_seq = make_sequences(X_val_ext, y_val_ext, seq_len)

    class_weight = get_class_weight_dict(ytr_seq) if use_class_weight else None

    model = build_lstm(
        input_shape=(seq_len, X_train.shape[1]),
        n_units=params["n_units"], dropout=params["dropout"],
        learning_rate=params["learning_rate"], seed=seed,
        n_layers=params.get("n_layers", 1),
    )
    model = _fit_with_checkpoint(
        model, Xtr_seq, ytr_seq, validation_data=(Xval_seq, yval_seq),
        epochs=epochs, batch_size=params["batch_size"],
        class_weight=class_weight, monitor="val_loss",
    )

    val_prob = model.predict(Xval_seq, verbose=0).ravel()
    best_thr, val_acc = find_best_threshold(yval_seq, val_prob)
    pred_rate_at_05 = float((val_prob >= 0.5).mean())

    train_prob = model.predict(Xtr_seq, verbose=0).ravel()
    train_preds = (train_prob >= best_thr).astype(int)
    train_acc = float((train_preds == ytr_seq).mean())

    return val_acc, train_acc, best_thr, pred_rate_at_05


def _run_one_candidate(params, source, X_train, y_train, X_val, y_val,
                        epochs, seed, use_class_weight):

    val_acc, train_acc, best_thr, pred_rate_at_05 = _evaluate_lstm_candidate(
        X_train, y_train, X_val, y_val, params, epochs, seed, use_class_weight,
    )
    return {
        **params, "threshold": best_thr, "source": source,
        "val_accuracy": float(val_acc), "train_accuracy": train_acc,
        "pred_rate_class1_at_0.5": pred_rate_at_05,
    }

def optimize_lstm(X_train, y_train, X_val, y_val, param_grid: dict,
                   max_combinations: int, epochs: int, seed: int,
                   default_params: dict | None = None,
                   use_class_weight: bool = True,
                   n_jobs: int = 1):
    
    if len(y_val) == 0:
        raise ValueError(
            "optimize_lstm zahtijeva neprazan validacioni skup "
            "(config.yaml: split.val_fraction mora biti > 0)."
        )

    rng = np.random.default_rng(seed)
    keys = list(param_grid.keys())
    all_combos = list(itertools.product(*param_grid.values()))
    rng.shuffle(all_combos)
    combos = all_combos[:max_combinations]

    candidates = [(dict(zip(keys, combo)), "grid_search") for combo in combos]
    if default_params is not None:
        default_hp = {k: v for k, v in default_params.items() if k != "threshold"}
        candidates.append((default_hp, "default"))

    if n_jobs == 1:
        
        log = [
            _run_one_candidate(params, source, X_train, y_train, X_val, y_val,
                                epochs, seed, use_class_weight)
            for params, source in candidates
        ]
    else:
        log = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_run_one_candidate)(
                params, source, X_train, y_train, X_val, y_val,
                epochs, seed, use_class_weight,
            )
            for params, source in candidates
        )

    best_entry = max(log, key=lambda entry: entry["val_accuracy"])
    meta_keys = ("val_accuracy", "train_accuracy", "pred_rate_class1_at_0.5")
    best_params = {k: v for k, v in best_entry.items() if k not in meta_keys}
    best_score = best_entry["val_accuracy"]
    best_train_accuracy = best_entry["train_accuracy"]

    return best_params, best_score, best_train_accuracy, log

def train_final_lstm_multi_run(X_trainval, y_trainval, X_test, y_test, best_params: dict,
                                n_runs: int, epochs: int, base_seed: int | None = None,
                                use_class_weight: bool = True):

    seq_len = best_params["sequence_length"]
    threshold = best_params.get("threshold", 0.5)

    Xtv_seq, ytv_seq = make_sequences(X_trainval, y_trainval, seq_len)
    X_test_ext, y_test_ext = _extend_for_sequence(X_trainval, y_trainval, X_test, y_test, seq_len)
    Xtest_seq, ytest_seq = make_sequences(X_test_ext, y_test_ext, seq_len)

    if len(ytest_seq) != len(y_test):
        raise ValueError(
            "LSTM test sekvence nisu usaglasene sa testnim skupom: "
            f"dobijeno {len(ytest_seq)} sekvenci, ocekivano {len(y_test)}. "
            "Provjeriti seq_len, X_test_ext i pravilo poravnanja targeta."
        )

    class_weight = get_class_weight_dict(ytv_seq) if use_class_weight else None

    results = []
    for run_i in range(n_runs):
        seed = (base_seed if base_seed is not None else 1000) + run_i
        model = build_lstm(
            input_shape=(seq_len, X_trainval.shape[1]),
            n_units=best_params["n_units"], dropout=best_params["dropout"],
            learning_rate=best_params["learning_rate"], seed=seed,
            n_layers=best_params.get("n_layers", 1),
        )
        model = _fit_with_checkpoint(
            model, Xtv_seq, ytv_seq, validation_data=None,
            epochs=epochs, batch_size=best_params["batch_size"],
            class_weight=class_weight, monitor="loss",
        )

        y_prob = model.predict(Xtest_seq, verbose=0).ravel()
        y_pred = (y_prob >= threshold).astype(int)
        acc = float((y_pred == ytest_seq).mean())
        pred_rate_class1 = float(y_pred.mean())

        results.append({
            "seed": seed, "accuracy": acc, "y_true": ytest_seq, "y_pred": y_pred,
            "n_test_sequences": len(ytest_seq), "threshold_used": threshold,
            "pred_rate_class1": pred_rate_class1,
        })
    return results