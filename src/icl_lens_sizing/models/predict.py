import statsmodels.api as sm
import pandas as pd


def predict_and_validate_linear_regression(X_validation, y_validation, est) -> pd.DataFrame:

    X_prime = sm.add_constant(X_validation, has_constant='add')
    prediction = est.predict(X_prime)

    column_names = {0: 'prediction', 1: 'truth'}
    df_validation = pd.DataFrame([prediction, y_validation]).transpose().rename(columns=column_names)
    df_validation["delta"] = df_validation["prediction"] - df_validation["truth"]

    return df_validation


def predict_and_validate_lasso_regression(X_validation, y_validation, model_lasso):

    preds = model_lasso.predict(X_validation)
    column_names = {0: 'prediction', 1: 'truth'}
    df_validation = pd.DataFrame([preds, y_validation]).transpose().rename(columns=column_names)
    df_validation["delta"] = df_validation["prediction"] - df_validation["truth"]

    return df_validation
