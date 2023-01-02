import statsmodels as sm
import pandas as pd


def predict_and_validate_linear_regression(X_validation, est, y_validation):

    X_prime = sm.add_constant(X_validation, has_constant='add')
    prediction = est.predict(X_prime)

    column_names = {0: 'prediction', 1: 'truth'}
    df_validation = pd.DataFrame([prediction, y_validation]).transpose().rename(columns=column_names)
    df_validation["delta"] = df_validation["prediction"] - df_validation["truth"]

    return df_validation
