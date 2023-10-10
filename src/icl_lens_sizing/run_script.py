from preprocessing.preprocessing import prepare_training_data, prepare_pred_data
import statsmodels.api as sm
from sklearn.linear_model import Lasso
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from src.icl_lens_sizing.reporting.reports import create_named_data_frames, prepare_dfs_for_pred, \
    predict_ols_for_all_items, predict_for_all_items


def run_all(write: bool = True):

    # Load data from csv sheet
    path = "/Users/sortmanns/git/work/icl-lens-sizing/data/icl_data_2023-07-09.csv"
    df = pd.read_csv(path, sep=";", decimal=',')

    # prepare data
    features = ['implantat_size', 'AtA', 'ACW', 'ARtAR_LR', 'StS', 'CBID', 'WtW_MS-39', 'Sphaere']
    custom_features = None # {'spherical_equivalent': (lambda row: row['Sphaere']-0.5*row['Zylinder'])}

    categoricals = None

    standart_scaler = StandardScaler()

    bid = 1
    predictions = pd.DataFrame()
    while bid <= df['befund_id'].max():

        if bid != 19:
            pred_df = df[(df['befund_id'] == bid) | (df['befund_id'] == bid + 1)]
            train_df = df[~((df['befund_id'] == bid) | (df['befund_id'] == bid + 1))]
            bid = bid+2
            implantate = [[11.25, 11.50, 11.75, 12.00, 12.25, 12.50, 12.75, 13.00, 13.25, 13.50, 13.75],
                          [11.25, 11.50, 11.75, 12.00, 12.25, 12.50, 12.75, 13.00, 13.25, 13.50, 13.75]]
        else:
            pred_df = df[df['befund_id']==bid]
            train_df = df[~(df['befund_id'] == bid)]
            bid=bid+1
            implantate = [[11.25, 11.50, 11.75, 12.00, 12.25, 12.50, 12.75, 13.00, 13.25, 13.50, 13.75]]

        pred_df['implantat_size'] = implantate
        pred_df = pred_df.explode(column='implantat_size')
        pred_df = pred_df.drop(columns=["Lens-ICPL-Distance","implantat_name",'auge',"befund_id"])
        pred_df = prepare_pred_data(pred_df, features=features, categorical_features=categoricals,
                                    custom_features=custom_features)


        feature_df, target_df, feat_map = prepare_training_data(train_df , target="Lens-ICPL-Distance",
                                                                                       features=features, categorical_features=categoricals,
                                                                                       custom_features=custom_features)

        # standart_scaler.fit(feature_df)
        # X_train = standart_scaler.transform(feature_df)
        # X_predict = standart_scaler.transform(pred_df[['implantat_size', 'AtA', 'ACW', 'ARtAR_LR', 'StS', 'CBID', 'WtW_MS-39', 'Sphaere']])
        # train linear regression
        # X_ols = sm.add_constant(X_train, has_constant='add')
        # est = sm.OLS(y_train, X_ols).fit()
        columns = list(feature_df.columns)
        X_train_df = feature_df
        X_predict_df = pred_df[['implantat_size', 'AtA', 'ACW', 'ARtAR_LR', 'StS', 'CBID', 'WtW_MS-39', 'Sphaere']]
        # train lasso regression
        model_lasso = Lasso()
        model_lasso.fit(X_train_df, target_df)

        preds_lasso = model_lasso.predict(X_predict_df)
        # preds_ols = est.predict(pred_df)
        X_predict_df['prediction'] = preds_lasso
        predictions = predictions.append(X_predict_df, ignore_index=False)


    if write:
        # Get the current date
        current_date = datetime.now()

        # Create the date string with underscores as separators
        date_string = current_date.strftime("%Y_%m_%d")

        # preds_ols.to_csv(f"../../docs/predictions/preds_ols_{date_string}.csv")
        predictions.to_csv(f"../../docs/predictions/preds_lasso_{date_string}.csv")

    return preds_lasso  #, preds_ols

run_all(write=True)
