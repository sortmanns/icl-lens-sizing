import pandas as pd
import numpy as np


def load_and_prepare_data(shuffle_rows=False, features=None, validation_size=0):
    # Load data from csv sheet
    df = pd.read_csv("../../data/Basismessungen-Table-1.csv", sep=";", header=1)

    # Drop artificial column
    df = df.drop("Unnamed: 0", axis=1)
    df = df.replace("-", None)

    df = df.drop(columns=["ID", "Eye", "ICL"])

    if features:
        cols = ["ICL-Size", "Vault"] + features
        df = df[cols]
    # df = df[["ICL-Size", "Vault", 'StS LR', 'CBID LR']]
    # df['ratio'] = df['CBID LR']/df['mPupil']
    # Drop rows containing NaN's
    df = df.dropna()

    # Swap first two columns
    feature_cols = list(df.columns)[2:]
    columns_titles = ["Vault", "ICL-Size"] + feature_cols
    df = df.reindex(columns=columns_titles)
    feature_mapping = {}
    for i, col in enumerate(df.columns):
        feature_mapping[i] = col

    print(feature_mapping)
    # Shuffle rows
    if shuffle_rows:
        df_lasso = df.reindex(np.random.permutation(df.index))
    else:
        df_lasso = df

    # Mark categorical features
    # df_lasso['ICL-Size'] = df_lasso['ICL-Size'].astype(str).astype('category')

    # Transform Pandas dataframes to NumPy arrays
    X = df_lasso.values[:, 1:19]
    y = df_lasso.values[:, 0]

    # Generate encoded categorical features from ICL-size
    # X_categorical, dictnames = sm.tools.categorical(X, col = 0, dictnames= True, drop=True)

    # Convert array type back to float
    X_categorical = np.array(X, dtype=float)

    print(f"Data shape: {X_categorical.shape}")
    # Split data into training and test data
    n_rows = X_categorical.shape[0]
    X_train = X_categorical[0:n_rows - validation_size, :]
    X_validation = X_categorical[n_rows - validation_size:n_rows, :]
    y_train = np.array(y[0:n_rows - validation_size], dtype=float)
    y_validation = np.array(y[n_rows - validation_size:n_rows], dtype=float)

    return X_train, y_train, X_validation, y_validation, df_lasso
