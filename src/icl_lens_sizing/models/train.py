import statsmodels as sm
import numpy as np


def train_linear_regression(X_train: np.array, y_train: np.array):

    X_ols = sm.add_constant(X_train, has_constant='add')
    est = sm.OLS(y_train, X_ols).fit()

    return est
