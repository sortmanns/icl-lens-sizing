from preprocessing.preprocessing import load_and_prepare_training_data, load_and_prepare_new_data
import statsmodels.api as sm
from sklearn.linear_model import Lasso

from src.icl_lens_sizing.reporting.reports import create_signed_data_frames, prepare_dfs_for_pred, \
    predict_ols_for_all_items, predict_for_all_items

# prepare data
features = ['ACD', 'ACA nasal', 'ACA temporal', 'AtA', 'ACW',
            'ARtARLR', 'StS', 'StS LR', 'CBID', 'CBID LR', 'mPupil', 'WtW MS-39',
            'WtW IOL Master', 'Sphäre', 'Zylinder', 'Sphärisches Äquivalent']

X_train, y_train, X_validation, y_validation, df = load_and_prepare_training_data(features=features, validation_size=0)

# train linear regression
X_ols = sm.add_constant(X_train, has_constant='add')
est = sm.OLS(y_train, X_ols).fit()

# train lasso regression
model_lasso = Lasso()
model_lasso.fit(X_train, y_train)

# predict for new data
df_test, feat_map = load_and_prepare_new_data(df)
selectors = ['Eye', 'ID']
df_dicti = create_signed_data_frames({'df': df_test}, selectors)

feat_map[0] = 'ICL-Size'
df_dict_ols = prepare_dfs_for_pred(df_dicti, ['ICL-Size'], feat_map, df[['ICL-Size']])

cat_map = {}
for i, cat in enumerate(df['ICL-Size'].unique()):
    cat_map[i] = cat

preds_ols = predict_ols_for_all_items(df_dict_ols, est, cat_map)

preds_lasso = predict_for_all_items(df_dict_ols, model_lasso, cat_map)

preds_ols.to_csv("../../docs/predictions/preds_ols_2023_01_02.csv")
preds_lasso.to_csv("../../docs/predictions/preds_lasso_2023_01_02.csv")
