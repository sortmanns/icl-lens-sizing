from preprocessing.preprocessing import prepare_training_data, prepare_pred_data
import statsmodels.api as sm
from sklearn.linear_model import Lasso
import pandas as pd

from src.icl_lens_sizing.reporting.reports import create_named_data_frames, prepare_dfs_for_pred, \
    predict_ols_for_all_items, predict_for_all_items


def run_all(write: bool = True):

    # Load data from csv sheet
    path = "/Users/sortmanns/git/work/icl-lens-sizing/data/icl_data_2023-07-09.csv"
    df = pd.read_csv(path, sep=";", decimal=',')

    # prepare data
    features = ['implantat_größe','geschlecht','alter', 'ACD', 'ACA_nasal', 'ACA_temporal', 'AtA', 'ACW',
                'ARtAR_LR', 'StS', 'StS_LR', 'CBID', 'CBID_LR', 'mPupil', 'WtW_MS-39',
                'WtW_IOL_Master', 'Sphaere', 'Zylinder', "Achse"]

    custom_features = {'cbid_ratio': (lambda row: row['CBID'] / row['CBID_LR']),
                       'spherical_equivalent': (lambda row: row['Sphaere']+0.5*row['Zylinder'])}

    categoricals = ['geschlecht']

    bid = 1
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

        pred_df['implantat_größe'] = implantate
        pred_df = pred_df.explode(column='implantat_größe')
        pred_df = pred_df.drop(columns=["Lens-ICPL-Distance","implantat_name",'auge',"befund_id"])
        pred_df = prepare_pred_data(pred_df, features=features, categorical_features=categoricals,
                                    custom_features=custom_features)


        X_train, y_train, X_validation, y_validation, feat_map = prepare_training_data(train_df , target="Lens-ICPL-Distance",
                                                                                       features=features, test_size=0.2, categorical_features=categoricals,
                                                                                       custom_features=custom_features, standardize=False)

        # train linear regression
        # X_ols = sm.add_constant(X_train, has_constant='add')
        # est = sm.OLS(y_train, X_ols).fit()

        # train lasso regression
        model_lasso = Lasso()
        model_lasso.fit(X_train, y_train)

        preds_lasso = model_lasso.predict(pred_df)
        # preds_ols = est.predict(pred_df)
        print(preds_lasso)

    if write:
        preds_ols.to_csv("../../docs/predictions/preds_ols_2023_04_20.csv")
        preds_lasso.to_csv("../../docs/predictions/preds_lasso_2023_04_20.csv")

    return preds_lasso  #, preds_ols

run_all(write=False)
