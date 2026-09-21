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


def _extract_actual_params(model_name: str, fitted_clf, param_grid_for_model: dict) -> dict:
 
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


def _find_early_stopped_n_estimators(model_name: str, X_train, y_train, X_val, y_val,
                                      best_params: dict, seed: int,
                                      max_n_estimators: int, early_stopping_rounds: int) -> int:

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

def train_all_classical_models(X_train, y_train, X_val, y_val, X_trainval, y_trainval,
                                X_test, y_test, param_grids: dict, n_cv_folds: int,
                                n_jobs: int, seed: int, gap: int = 1,
                                early_stopping_cfg: dict | None = None) -> dict:

    if len(y_val) == 0:
        raise ValueError(
            "train_all_classical_models zahtijeva neprazan validacioni skup "
            "(config.yaml: split.val_fraction mora biti > 0)."
        )

    early_stopping_cfg = early_stopping_cfg or {"max_n_estimators": 1000, "early_stopping_rounds": 20}

    results = {
        "baseline_predictions": {},   
        "optimized_predictions": {}, 
        "train_predictions": {},      
        "val_predictions": {},        
        "best_params": {},
        "early_stopping_info": {},    
        "model_selection": {},        
        "final_params_used": {},     
    }

    for name in param_grids:
        pipe_default_tr = make_pipeline(name, seed)
        pipe_default_tr.fit(X_train, y_train)
        default_val_acc = accuracy_score(y_val, pipe_default_tr.predict(X_val))

        pipe_default_trainval = make_pipeline(name, seed)
        pipe_default_trainval.fit(X_trainval, y_trainval)
        results["baseline_predictions"][name] = pipe_default_trainval.predict(X_test)

        search = GridSearchCV(
            make_pipeline(name, seed), param_grids[name],
            cv=TimeSeriesSplit(n_splits=n_cv_folds, gap=gap),
            scoring="accuracy", n_jobs=n_jobs, refit=True,
        )
        search.fit(X_train, y_train)
        best_params = dict(search.best_params_)

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
            "selected": selected,
        }

        results["train_predictions"][name] = selected_train_pipe.predict(X_train)
        results["val_predictions"][name] = selected_train_pipe.predict(X_val)

        final_pipe = make_pipeline(name, seed)
        if selected_params:
            final_pipe.set_params(**selected_params)
        final_pipe.fit(X_trainval, y_trainval)
        results["optimized_predictions"][name] = final_pipe.predict(X_test)

       
        results["final_params_used"][name] = _extract_actual_params(
            name, final_pipe.named_steps["clf"], param_grids[name]
        )

    return results