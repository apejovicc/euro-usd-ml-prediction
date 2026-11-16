"""
LSTM model - poravnanje zadatka i stvarna optimizacija hiperparametara.

Mentorska tacka 5: sekvenca duzine L koja se zavrsava u danu t predvidja
ISTI target(t) = 1[Close(t+1) > Close(t)] kao sto koriste klasicni
modeli u trenutku t (make_sequences implementira ovo eksplicitno).

Mentorska tacka 6: stvarna optimizacija (units, dropout, learning rate,
duzina sekvence, batch_size, broj slojeva) ISKLJUCIVO na validacionom
skupu, sa stvarno implementiranim EarlyStopping, ReduceLROnPlateau i
ModelCheckpoint (tezine se STVARNO ucitavaju nazad nakon treninga, ne
samo cuvaju i brisu - vidi _fit_with_checkpoint).

Dijagnostika kolapsa u skoro-konstantnu predikciju:
- class_weight: tezine klasa (balanced) racunaju se iz trening skupa.
- threshold tuning: prag odluke se bira na VALIDACIONOM skupu.

optimize_lstm vraca i train_accuracy pobjednicke konfiguracije iz FAZE
IZBORA (racunatu na X_train, istim modelom koji je ocijenjen na X_val) -
koristi se samo za izvjestavanje o izboru. Generalizacioni jaz se racuna
na KONACNOM modelu (train_final_lstm_multi_run): in-sample tacnost na
D_trainval i tacnost na D_test, za svako od n_runs pokretanja.

Ogranicenja procedure izbora (navode se u radu): svaki kandidat se trenira
jednom (jedan seed), pa je validaciona tacnost kandidata sama po sebi sum;
prag odluke se bira na istom validacionom skupu kao i konfiguracija, pa je
maksimalna validaciona tacnost optimisticno pristrasna. Zbog toga log cuva
i validacionu tacnost pri fiksnom pragu 0.5.

NAPOMENA: funkcije ovdje pretpostavljaju da POSTOJI zaseban validacioni
skup (X_val/y_val), u skladu sa config.yaml: split.val_fraction=0.15.
"""
from __future__ import annotations
import itertools
import os
import tempfile
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.utils.class_weight import compute_class_weight


# ---------------------------------------------------------------------
# make_sequences(): gradi LSTM sekvence od niza karakteristika, tako da
# sekvenca koja se zavrsava u danu t predvidja ISTI target(t) koji
# koriste i klasicni modeli. Baca gresku ako su X i y neusaglaseni.
# ---------------------------------------------------------------------
def make_sequences(X: np.ndarray, y: np.ndarray, seq_len: int):
    """Gradi sekvence za LSTM tako da svaka sekvenca X_seq[i] zavrsava u
    istom vremenskom trenutku t u kojem klasicni modeli koriste feature
    vektor za predikciju targeta t = 1[Close(t+1) > Close(t)].

    Precizno:
      X_seq[i] = X[i : i + seq_len]
      y_seq[i] = y[i + seq_len - 1]
    """
    n = len(X) - seq_len + 1
    if n <= 0:
        raise ValueError("Duzina sekvence veca je od raspolozivog broja uzoraka.")
    if len(X) != len(y):
        raise ValueError("X i y moraju imati isti broj uzoraka da bi sekvence bile usaglasene.")
    X_seq = np.stack([X[i:i + seq_len] for i in range(n)])
    y_seq = y[seq_len - 1: seq_len - 1 + n]
    return X_seq, y_seq


# ---------------------------------------------------------------------
# get_class_weight_dict(): racuna balansirane tezine klasa iz trening
# skupa - koristi se u model.fit(class_weight=...) da funkcija gubitka
# podjednako kaznjava greske na obje klase.
# ---------------------------------------------------------------------
def get_class_weight_dict(y: np.ndarray) -> dict:
    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    return {int(c): float(w) for c, w in zip(classes, weights)}


# ---------------------------------------------------------------------
# find_best_threshold(): sweep praga odluke (0.30-0.70) na VALIDACIONOM
# skupu, umjesto fiksnog 0.5 - bira prag koji maksimizuje tacnost.
# ---------------------------------------------------------------------
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


# ---------------------------------------------------------------------
# build_lstm(): gradi Keras Sequential model sa n_layers LSTM slojeva
# (svi osim poslednjeg vracaju cijelu sekvencu da bi mogli da se stekuju)
# + Dense(1, sigmoid) izlazni sloj za binarnu klasifikaciju.
# ---------------------------------------------------------------------
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


# ---------------------------------------------------------------------
# _extend_for_sequence(): "pozajmljuje" poslednjih (seq_len-1) redova iz
# trening skupa i lijepi ih na pocetak val/test skupa, da prva sekvenca
# u tom skupu ima punu istoriju bez gubitka uzoraka na granici.
# ---------------------------------------------------------------------
def _extend_for_sequence(X_train, y_train, X_other, y_other, seq_len):
    """Dodaje posljednjih (seq_len-1) trening uzoraka na pocetak drugog
    skupa radi korektnog formiranja prvih LSTM sekvenci na granici."""
    if seq_len <= 1:
        return X_other, y_other
    X_ext = np.concatenate([X_train[-(seq_len - 1):], X_other])
    y_ext = np.concatenate([y_train[-(seq_len - 1):], y_other])
    return X_ext, y_ext


# ---------------------------------------------------------------------
# _fit_with_checkpoint(): trenira model uz sva tri callback-a
# (EarlyStopping, ReduceLROnPlateau, ModelCheckpoint). ModelCheckpoint
# sada STVARNO ucitava sacuvane najbolje tezine nazad u model (ne samo
# ih snima i brise) - mentorska tacka 6.
# ---------------------------------------------------------------------
def _fit_with_checkpoint(model, X_fit, y_fit, validation_data=None, *, epochs, batch_size,
                          class_weight, monitor: str):
    """Trenira model uz EarlyStopping + ReduceLROnPlateau + ModelCheckpoint.
    Za razliku od prethodne verzije, tezine sacuvane u checkpoint-u se
    STVARNO ucitavaju nazad u model nakon treninga (mentorska tacka 6 -
    'ili implementirati stvarno, ili ne tvrditi da se koristi'). Ovo je
    donekle redundantno sa EarlyStopping(restore_best_weights=True), ali
    oba mehanizma su ovdje stvarno funkcionalna, ne samo deklarativna.
    """
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
        model.load_weights(checkpoint_path)  # STVARNO ucitava najbolje tezine
        os.remove(checkpoint_path)

    return model

def _evaluate_lstm_candidate(X_train, y_train, X_val, y_val, params: dict,
                              epochs: int, seed: int, use_class_weight: bool):
    """Trenira i evaluira JEDNU LSTM konfiguraciju na validaciji. Vraca
    (val_acc, train_acc, best_threshold, pred_rate_at_05, val_acc_at_05). Koristi se i
    za kandidate iz grid pretrage i za fiksnu (default) konfiguraciju -
    obje prolaze kroz ISTU proceduru, da bi rezultati bili direktno
    uporedivi (mentorska tacka 8 - isti kriterijum za sve kandidate)."""
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
    val_acc_at_05 = float(((val_prob >= 0.5).astype(int) == yval_seq).mean())

    train_prob = model.predict(Xtr_seq, verbose=0).ravel()
    train_preds = (train_prob >= best_thr).astype(int)
    train_acc = float((train_preds == ytr_seq).mean())

    return val_acc, train_acc, best_thr, pred_rate_at_05, val_acc_at_05
# ---------------------------------------------------------------------
# optimize_lstm(): GLAVNA funkcija pretrage hiperparametara. Nasumicno
# bira max_combinations kombinacija iz param_grid, trenira svaku SAMO
# na X_train, evaluira ISKLJUCIVO na X_val (test skup se ovdje ne
# koristi), i vraca pobjednicku konfiguraciju + log svih pokusaja.
# ---------------------------------------------------------------------


def optimize_lstm(X_train, y_train, X_val, y_val, param_grid: dict,
                   max_combinations: int, epochs: int, seed: int,
                   default_params: dict | None = None,
                   use_class_weight: bool = True):
    """Random search nad param_grid, evaluacija ISKLJUCIVO na
    validacionom skupu (izbor arhitekture i praga). Za pobjednicku
    konfiguraciju dodatno se racuna i tacnost NA TRENING SKUPU (H3 -
    generalizacioni jaz), ali train_accuracy se NIKAD ne koristi za sam
    izbor hiperparametara - samo se izvjestava.

    Mentorska tacka 8 (kriterijum rangiranja - dosljednost sa klasicnim
    modelima): ako je proslijedjen default_params (fiksna konfiguracija
    iz config.yaml: lstm.default_hyperparameters), ta konfiguracija se
    evaluira NA ISTI NACIN kao i svaka grid-search kombinacija i
    RAVNOPRAVNO se takmici za najbolji rezultat - ne koristi se samo kao
    referentna tacka za H2 poredjenje, vec moze i da POBIJEDI grid search
    ako ima bolju validacionu tacnost (isti princip kao za XGBoost/
    LightGBM/SVM/LogReg u train_all_classical_models).

    Vraca (best_params, best_val_accuracy, best_train_accuracy, log).
    best_params sadrzi i kljuceve 'threshold' i 'source' ('grid_search'
    ili 'default').
    """
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

    log = []
    best_score = -np.inf
    best_params = None
    best_train_accuracy = None

    for params, source in candidates:
        val_acc, train_acc, best_thr, pred_rate_at_05, val_acc_at_05 = _evaluate_lstm_candidate(
            X_train, y_train, X_val, y_val, params, epochs, seed, use_class_weight,
        )

        params_with_threshold = {**params, "threshold": best_thr, "source": source}
        log.append({**params_with_threshold, "val_accuracy": float(val_acc),
                    "val_accuracy_at_0.5": val_acc_at_05,
                    "train_accuracy": train_acc,
                    "pred_rate_class1_at_0.5": pred_rate_at_05})

        if val_acc > best_score:
            best_score = val_acc
            best_params = params_with_threshold
            best_train_accuracy = train_acc

    return best_params, best_score, best_train_accuracy, log


# ---------------------------------------------------------------------
# train_final_lstm_multi_run(): trenira FINALNI model (sa vec izabranim
# best_params) N_STOCHASTIC_RUNS puta na train+val, testira na test
# skupu. Vise pokretanja jer je LSTM osjetljiv na slucajnu inicijalizaciju
# - izvjestavamo mean +/- std svih pokretanja. Pokretanje koje se koristi za
# matrice konfuzije i testove bira se unaprijed zadatim pravilom, bez
# upotrebe testnog skupa (vidi pipelines.py).
# ---------------------------------------------------------------------
def train_final_lstm_multi_run(X_trainval, y_trainval, X_test, y_test, best_params: dict,
                                n_runs: int, epochs: int, base_seed: int | None = None,
                                use_class_weight: bool = True):
    """Trenira konacni LSTM sa best_params n_runs puta, razlicitim
    seedovima. Vraca listu recnika po pokretanju; za SVAKO pokretanje se
    istim modelom racunaju tacnost na D_trainval (in-sample) i na D_test.

    Provjerava da broj testnih sekvenci odgovara velicini test skupa
    (mentorska tacka 5) - ako ne, baca gresku umjesto tihog neslaganja.
    """
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

        # Isti model, in-sample: tacnost na sekvencama iz D_trainval
        tv_prob = model.predict(Xtv_seq, verbose=0).ravel()
        trainval_acc = float(((tv_prob >= threshold).astype(int) == ytv_seq).mean())

        results.append({
            "seed": seed, "accuracy": acc, "y_true": ytest_seq, "y_pred": y_pred,
            "n_test_sequences": len(ytest_seq), "threshold_used": threshold,
            "pred_rate_class1": pred_rate_class1,
            "trainval_accuracy": trainval_acc,
            "n_trainval_sequences": len(ytv_seq),
        })
    return results