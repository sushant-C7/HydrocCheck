import numpy as np
def add_ratios(X):
    X = np.asarray(X)
    hard_ph = X[:, 1] / (X[:, 0] + 0.001)     # Hardness / pH
    org_chl = X[:, 6] * X[:, 3]               # Organic_carbon * Chloramines
    return np.column_stack([X, hard_ph, org_chl])