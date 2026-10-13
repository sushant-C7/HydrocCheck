import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from .config import *
from .pipeline import make_pipeline
df = pd.read_csv(DATA)
X, y = df[FEATURES], df[TARGET]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
X_tr.to_pickle(MODELS/"X_tr.pkl"); y_tr.to_pickle(MODELS/"y_tr.pkl")
X_te.to_pickle(MODELS/"X_te.pkl"); y_te.to_pickle(MODELS/"y_te.pkl")
models = {
  "LogReg": make_pipeline(LogisticRegression(max_iter=1000), smote=False),
  "RandomForest": make_pipeline(RandomForestClassifier(300, random_state=SEED), smote=False),
  "XGB_default": make_pipeline(XGBClassifier(eval_metric="logloss", random_state=SEED), smote=False),
  "XGB_cost_w2.5": make_pipeline(XGBClassifier(scale_pos_weight=2.5, eval_metric="logloss",
      random_state=SEED)),
}
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
rows = []
for name, m in models.items():
    r = cross_validate(m, X_tr, y_tr, cv=cv,
        scoring=["accuracy","precision","recall","f1","average_precision"])
    rows.append({"model": name, **{k[5:]: v.mean().round(3) for k, v in r.items() if
        k.startswith("test_")}})
res = pd.DataFrame(rows); print(res)
res.to_csv(REPORTS/"baselines.csv", index=False)