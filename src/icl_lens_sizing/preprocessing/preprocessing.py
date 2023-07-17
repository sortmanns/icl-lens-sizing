import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def prepare_training_data(df, target, features=None, categorical_features=None, standardize=True, test_size=0.2,
                          custom_features:dict=None):

    # Drop artificial column
    df = df.drop(columns=["implantat_name","auge"])

    if features:
        cols = [target] + features
        df = df[cols]

    # Drop rows containing NaN's
    df = df.dropna()

    # Split df into X and y
    target_df = df[target]
    feature_df = df.drop(columns=[target])


    if categorical_features:
        feature_df = pd.get_dummies(data=feature_df, columns=categorical_features, drop_first=True)

    if custom_features:
        for key, func in custom_features.items():
            feature_df[key] = feature_df.apply(func, axis=1)

    # Create feature mapping
    feature_mapping = {}
    for i, col in enumerate(feature_df.columns):
        feature_mapping[i] = col

    X_train, X_validation, y_train, y_validation = train_test_split(feature_df, target_df, test_size=test_size)
    if standardize:
        standart_scaler = StandardScaler()
        X_train = standart_scaler.fit_transform(X_train)
        X_validation = standart_scaler.fit_transform(X_validation)

    return X_train, y_train, X_validation, y_validation, feature_mapping


def prepare_pred_data(df, features=None, categorical_features=None, custom_features:dict=None):


    # Drop rows containing NaN's
    df = df.dropna()

    if categorical_features:
        feature_df = pd.get_dummies(data=df, columns=categorical_features, drop_first=True)

    if custom_features:
        for key, func in custom_features.items():
            feature_df[key] = feature_df.apply(func, axis=1)

    return feature_df
