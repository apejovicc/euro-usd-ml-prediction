"""
Klasicni modeli (logisticka regresija, SVM, XGBoost, LightGBM).

Mentorska tacka 4: StandardScaler je UNUTAR sklearn Pipeline-a, tako da
se u svakom CV foldu fituje iskljucivo na trening dio tog folda.

Mentorska tacka 6: prostor pretrage za svaki model dolazi iz
config.yaml (classical_models.param_grids) - JEDINI izvor koji se
koristi i ovdje za treniranje i u tekstu rada za Tabelu 3.1.

Mentorska tacka 6 (early stopping): za XGBoost i LightGBM, n_estimators
NIJE u GridSearchCV prostoru pretrage - umjesto toga se odredjuje
STVARNIM early stopping-om (eval_set=X_val, ne test skup).

Mentorska tacka 8 (kriterijum rangiranja): za svaki model se porede
FABRICKA (default) i GRID SEARCH konfiguracija NA VALIDACIJI (nikad na
testu), i bira se ona sa boljom validacionom tacnoscu. Ovo sprjecava
situaciju iz originalne verzije rada u kojoj je fabricki XGBoost imao
bolju testnu tacnost od "optimizovane" verzije, a "optimizovana" je
ipak automatski proglasena finalnom bez objasnjenja.

Generalizacioni jaz (jedinstvena definicija): mjeri se na ISTOM modelu -
konacnom modelu koji je fitovan na D_trainval = D_train + D_val. Za taj
model se racunaju tacnost na D_trainval (tacnost na podacima na kojima je
model ucen) i tacnost na D_test (podaci koje model nije vidio):
    gap = Acc(D_trainval) - Acc(D_test).
Predikcije modela iz FAZE IZBORA (fit samo na D_train, evaluiran na D_val;
to je model ciji se rezultat koristi u kriterijumu izbora) cuvaju se
odvojeno ("selection_*"), samo za izvjestavanje o izboru konfiguracije.

NAPOMENA: funkcije ovdje pretpostavljaju da POSTOJI zaseban validacioni
skup (X_val/y_val), u skladu sa config.yaml: split.val_fraction=0.15.

Napomena (Windows): n_jobs iz config.yaml se podrazumijevano postavlja
na 1 zbog poznatog konflikta joblib multiprocessing-a kada na sistemu
postoji vise Python instalacija (npr. Anaconda + venv).
"""
from __future__ import annotations
import lightgbm as lgb
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

EARLY_STOPPING_MODELS = ("xgboost", "lightgbm")


# ---------------------------------------------------------------------
# _extract_actual_params(): nakon fitovanja, cita STVARNE vrijednosti
# hiperparametara direktno iz modela (ne iz onoga sto smo mu proslijedili).
# Za XGBoost, koristi booster config jer get_params() vraca None za
# fabricke vrijednosti koje C++ biblioteka interno primjenjuje.
# ---------------------------------------------------------------------
def _extract_actual_params(model_name: str, fitted_clf, param_grid_for_model: dict) -> dict:
    """Vraca STVARNO rezolvovane vrijednosti hiperparametara nakon
    fitovanja. Za XGBoost, get_params() vraca None za neeksplicitno
    postavljene parametre (interni C++ default se tada primjenjuje bez
    da se vidi u Python objektu) - ovdje se ti default-i eksplicitno
    ocitavaju iz istreniranog modela (booster config), da bi tekst rada
    mogao tacno da navede koje su vrijednosti STVARNO koriscene, cak i
    kada je izabrana fabricka (default) konfiguracija (mentorska tacka 8).
    """
    keys = [k.replace("clf__", "") for k in param_grid_for_model.keys()]
    result = {}

    if model_name == "xgboost":
        import json
        booster_cfg = json.loads(fitted_clf.get_booster().save_config())
        tree_params = booster_cfg["learner"]["gradient_booster"]["tree_train_param"]
        for k in keys:
            if k in tree_params:
                val = tree_params[k]
                result[f"clf__{k}"] = int(val) if k == "max_depth" else float(val)
        result["clf__n_estimators"] = int(fitted_clf.get_booster().num_boosted_rounds())
    else:
        params = fitted_clf.get_params()
        for k in keys:
            result[f"clf__{k}"] = params.get(k)
        if model_name == "lightgbm":
            result["clf__n_estimators"] = params.get("n_estimators")

    return result


# ---------------------------------------------------------------------
# make_pipeline(): gradi Pipeline(StandardScaler + model) za jedan od
# 4 podrzana modela. Scaler unutar Pipeline-a garantuje da se u CV
# foldu fituje samo na trening dio tog folda (mentorska tacka 4).
# ---------------------------------------------------------------------
def make_pipeline(model_name: str, seed: int) -> Pipeline:
    if model_name == "logreg":
        clf = LogisticRegression(max_iter=2000, random_state=seed)
    elif model_name == "svm":
        clf = SVC(random_state=seed)
    elif model_name == "xgboost":
        clf = XGBClassifier(eval_metric="logloss", random_state=seed, n_jobs=-1)
    elif model_name == "lightgbm":
        clf = LGBMClassifier(random_state=seed, n_jobs=-1, verbose=-1)
    else:
        raise ValueError(f"Nepoznat model: {model_name}")

    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", clf),
    ])


# ---------------------------------------------------------------------
# _find_early_stopped_n_estimators(): za XGBoost/LightGBM, umjesto da
# n_estimators trazimo Grid Search-om, treniramo sa velikim gornjim
# limitom (npr. 1000) i pustamo da se trening sam zaustavi kad
# validacija prestane da se poboljsava.
# ---------------------------------------------------------------------
def _find_early_stopped_n_estimators(model_name: str, X_train, y_train, X_val, y_val,
                                      best_params: dict, seed: int,
                                      max_n_estimators: int, early_stopping_rounds: int) -> int:
    """Fituje model SAMO na trening dijelu skupa, koristeci scaler i val
    skup kao eval_set za early stopping. Skaliranje se izvodi ISKLJUCIVO
    na trening podacima, a validacioni skup se transformise preko istog
    vec fitovanog scaler-a (nema curenja informacija kroz scaler).
    """
    pipe = make_pipeline(model_name, seed)
    X_train_scaled = pipe.named_steps["scaler"].fit_transform(X_train)
    X_val_scaled = pipe.named_steps["scaler"].transform(X_val)

    max_depth = best_params["clf__max_depth"]
    learning_rate = best_params["clf__learning_rate"]

    if model_name == "xgboost":
        clf = XGBClassifier(
            eval_metric="logloss", random_state=seed, n_jobs=-1,
            n_estimators=max_n_estimators, early_stopping_rounds=early_stopping_rounds,
            max_depth=max_depth, learning_rate=learning_rate,
        )
        clf.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)], verbose=False)
        return int(clf.best_iteration) + 1

    elif model_name == "lightgbm":
        clf = LGBMClassifier(
            random_state=seed, n_jobs=-1, verbose=-1, n_estimators=max_n_estimators,
            max_depth=max_depth, learning_rate=learning_rate,
        )
        clf.fit(X_train_scaled, y_train, eval_X=X_val_scaled, eval_y=y_val,
                callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)])
        return int(clf.best_iteration_)

    raise ValueError(f"Early stopping nije definisan za: {model_name}")


# ---------------------------------------------------------------------
# train_all_classical_models(): GLAVNA funkcija. Za svaki od 4 modela:
# 1) trenira fabricku (default) konfiguraciju, mjeri tacnost na validaciji
# 2) trenira Grid Search konfiguraciju (+ early stopping za XGB/LGBM), mjeri tacnost na validaciji
# 3) BIRA bolju od te dvije (kriterijum rangiranja - mentorska tacka 8)
# 4) fituje izabranu konfiguraciju na train+val, testira na test skupu
# 5) biljezi train/val/test predikcije i stvarno koriscene hiperparametre
# ---------------------------------------------------------------------
def train_all_classical_models(X_train, y_train, X_val, y_val, X_trainval, y_trainval,
                                X_test, y_test, param_grids: dict, n_cv_folds: int,
                                n_jobs: int, seed: int, gap: int = 1,
                                early_stopping_cfg: dict | None = None) -> dict:
    """Trenira sve klasicne modele, poredi fabricku i Grid Search
    konfiguraciju NA VALIDACIJI (mentorska tacka 8), i bira bolju za
    finalnu test evaluaciju. Vraca recnik sa svim predikcijama,
    hiperparametrima i odlukom o izboru konfiguracije - jedinstven
    izvor istine (mentorska tacka 7).
    """
    if len(y_val) == 0:
        raise ValueError(
            "train_all_classical_models zahtijeva neprazan validacioni skup "
            "(config.yaml: split.val_fraction mora biti > 0)."
        )

    early_stopping_cfg = early_stopping_cfg or {"max_n_estimators": 1000, "early_stopping_rounds": 20}

    results = {
        "baseline_predictions": {},   # fabricka konfig. (fit na train+val), test skup
        "optimized_predictions": {},  # KONACNO IZABRANA konfig. (fit na train+val), test skup
        "trainval_predictions": {},   # isti konacni model, predikcije na train+val (in-sample)
        "selection_train_predictions": {},  # model iz faze izbora (fit na train), skup train
        "selection_val_predictions": {},    # isti model iz faze izbora, skup val
        "best_params": {},            # sta je Grid Search nasao (bez obzira da li je izabran)
        "early_stopping_info": {},    # samo za xgboost/lightgbm
        "model_selection": {},        # kriterijum rangiranja - mentorska tacka 8
        "final_params_used": {},      # STVARNO koriscene vrijednosti (ukljucujuci fabricke default-e)
    }

    for name in param_grids:
        # ---- Fabricka (default) konfiguracija: fit na train, eval na val ----
        pipe_default_tr = make_pipeline(name, seed)
        pipe_default_tr.fit(X_train, y_train)
        default_val_acc = accuracy_score(y_val, pipe_default_tr.predict(X_val))

        # Fabricka konfig. na train+val, za slucaj da bude izabrana kao finalna
        pipe_default_trainval = make_pipeline(name, seed)
        pipe_default_trainval.fit(X_trainval, y_trainval)
        results["baseline_predictions"][name] = pipe_default_trainval.predict(X_test)

        # ---- Grid Search konfiguracija -----------------------------------
        search = GridSearchCV(
            make_pipeline(name, seed), param_grids[name],
            cv=TimeSeriesSplit(n_splits=n_cv_folds, gap=gap),
            scoring="accuracy", n_jobs=n_jobs, refit=True,
        )
        search.fit(X_train, y_train)
        best_params = dict(search.best_params_)
        cv_best_accuracy = float(search.best_score_)
        cv_best_std = float(search.cv_results_["std_test_score"][search.best_index_])

        if name in EARLY_STOPPING_MODELS:
            n_est = _find_early_stopped_n_estimators(
                name, X_train, y_train, X_val, y_val, best_params, seed,
                early_stopping_cfg["max_n_estimators"], early_stopping_cfg["early_stopping_rounds"],
            )
            best_params["clf__n_estimators"] = n_est
            results["early_stopping_info"][name] = {
                "n_estimators_selected": n_est,
                "max_n_estimators": early_stopping_cfg["max_n_estimators"],
                "early_stopping_rounds": early_stopping_cfg["early_stopping_rounds"],
            }
        results["best_params"][name] = best_params

        grid_search_pipe_tr = make_pipeline(name, seed)
        grid_search_pipe_tr.set_params(**best_params)
        grid_search_pipe_tr.fit(X_train, y_train)
        grid_search_val_acc = accuracy_score(y_val, grid_search_pipe_tr.predict(X_val))

        # ---- KRITERIJUM RANGIRANJA (mentorska tacka 8) ---------------------
        # Bira se konfiguracija sa boljom VALIDACIONOM tacnoscu - test skup
        # se ovdje uopste ne koristi za odluku.
        if grid_search_val_acc >= default_val_acc:
            selected = "grid_search"
            selected_params = best_params
            selected_train_pipe = grid_search_pipe_tr
        else:
            selected = "default"
            selected_params = {}
            selected_train_pipe = pipe_default_tr

        results["model_selection"][name] = {
            "default_val_accuracy": round(float(default_val_acc), 4),
            "grid_search_val_accuracy": round(float(grid_search_val_acc), 4),
            "grid_search_cv_accuracy": round(cv_best_accuracy, 4),
            "grid_search_cv_std": round(cv_best_std, 4),
            "selected": selected,
        }

        # Model iz faze izbora (fit samo na D_train) - samo za izvjestavanje
        results["selection_train_predictions"][name] = selected_train_pipe.predict(X_train)
        results["selection_val_predictions"][name] = selected_train_pipe.predict(X_val)

        # Finalni test: nov pipeline, fitovan na train+val, sa IZABRANOM konfig.
        final_pipe = make_pipeline(name, seed)
        if selected_params:
            final_pipe.set_params(**selected_params)
        final_pipe.fit(X_trainval, y_trainval)
        results["optimized_predictions"][name] = final_pipe.predict(X_test)
        # Isti konacni model: in-sample predikcije (za generalizacioni jaz)
        results["trainval_predictions"][name] = final_pipe.predict(X_trainval)

        # Stvarno koriscene vrijednosti hiperparametara (mentorska tacka 8) -
        # cak i kada je "selected"="default", ovdje se eksplicitno biljeze
        # prave vrijednosti (npr. XGBoost max_depth=6), ne prazan recnik.
        results["final_params_used"][name] = _extract_actual_params(
            name, final_pipe.named_steps["clf"], param_grids[name]
        )

    return results