import json, optuna, pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier
from .config import *
from .pipeline import make_pipeline
X_tr = pd.read_pickle(MODELS/"X_tr.pkl"); y_tr = pd.read_pickle(MODELS/"y_tr.pkl")
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
def objective(t):
    clf = XGBClassifier(
        max_depth=t.suggest_int("max_depth", 3, 10),
        learning_rate=t.suggest_float("learning_rate", 0.01, 0.2, log=True),
        subsample=t.suggest_float("subsample", 0.6, 1.0),
        reg_lambda=t.suggest_float("reg_lambda", 1e-2, 10, log=True),
        reg_alpha=t.suggest_float("reg_alpha", 1e-3, 10, log=True),
        n_estimators=t.suggest_int("n_estimators", 100, 500),
        scale_pos_weight=2.5, eval_metric="logloss", random_state=SEED)
    return cross_val_score(make_pipeline(clf), X_tr, y_tr, cv=cv, scoring="f1").mean()
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective, n_trials=50)
json.dump(study.best_params, open(MODELS/"best_params.json", "w"), indent=2)
print(study.best_value, study.best_params)