import pandas as pd
import numpy as np

def prepare_training_data(df, target, features=None, categorical_features=None, custom_features:dict=None):

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

    return feature_df, target_df, feature_mapping


def prepare_pred_data(df, features=None, categorical_features=None, custom_features:dict=None):


    # try:
    #     geschlecht = df["geschlecht"].iloc[0]
    # except:
    #     geschlecht = None
    #
    # if geschlecht:
    #     if geschlecht=='w':
    #         df['geschlecht_w']=1
    #         df = df.drop(columns='geschlecht')
    #     else:
    #         df['geschlecht_w'] = 0
    #         df = df.drop(columns='geschlecht')


    if categorical_features:
        feature_df = pd.get_dummies(data=df, columns=categorical_features, drop_first=False)
    else:
        feature_df = df

    try:
        feature_df["geschlecht_w"] = feature_df["geschlecht_m"]-1
        feature_df = feature_df.drop(columns=['geschlecht_m'])
    except:
        pass

    try:
        feature_df["geschlecht_w"] = feature_df["geschlecht_div"]-1
        feature_df = feature_df.drop(columns=['geschlecht_div'])
    except:
        pass

    if custom_features:
        for key, func in custom_features.items():
            feature_df[key] = feature_df.apply(func, axis=1)

    if feature_df["alter"].isna().sum() > 0:
        feature_df["alter"] = 40
    feature_df = feature_df.drop(columns=['geschlecht'])

    return feature_df
