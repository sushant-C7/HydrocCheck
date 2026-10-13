from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import RobustScaler, FunctionTransformer
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.pipeline import Pipeline
from .features import add_ratios
from .config import SEED
def make_pipeline(clf, smote=True):
    steps = [
        ("impute", IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=SEED)),
        ("ratios", FunctionTransformer(add_ratios)),
        ("scale", RobustScaler()),
    ]
    if smote:
        steps.append(("smote", BorderlineSMOTE(random_state=SEED)))
    steps.append(("clf", clf))
    return Pipeline(steps)