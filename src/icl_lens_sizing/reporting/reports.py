import pandas as pd
import statsmodels.api as sm


def create_signed_data_frames(df_dict: dict, selectors: list) -> dict:
    if not selectors:

        return df_dict

    else:
        sel = selectors.pop(0)
        df_dict_suc = creator_helper(df_dict, sel)

        return create_signed_data_frames(df_dict_suc, selectors)


def creator_helper(df_dict_2, sel):
    df_dictus = {}
    for key, df in df_dict_2.items():

        identifiers = df[sel].unique()
        for ident in identifiers:
            df_dictus[f"{key}_{sel}_{ident}"] = df[df[sel] == ident].drop(columns=sel)

    return df_dictus


def prepare_dfs_for_pred(df_dict: dict, categorical_cols: list, feat_map: dict,
                         df_add: pd.DataFrame = None, to_array: bool = False,
                         get_dummies: bool = False):
    new_dict = {}
    for key, df in df_dict.items():

        for cat_col in categorical_cols:
            df = categorize(df, df_add, cat_col, feat_map, get_dummies)

        if to_array:
            new_dict[key] = df.values
        else:
            new_dict[key] = df

    return new_dict


def categorize(df, df_add, cat_col, feat_map, get_dummies):
    if df_add.empty:
        cats = df_add[f'{cat_col}'].unique()
    else:
        cats = df_add[cat_col].unique()

    cats = [str(c) for c in cats]
    n = len(cats)
    df = df.append([df] * (n - 1), ignore_index=True)
    num_rows = df.shape[0]

    cat_rows = []
    for cat in cats:
        cat_rows.append([cat])

    df_cats = pd.DataFrame(cat_rows, columns=[cat_col])
    df_extended = pd.concat([df_cats, df], axis=1, ignore_index=True)
    df_extended = df_extended.rename(mapper=feat_map, axis=1)

    try:
        df_extended[cat_col] = df_extended[cat_col].astype(float)
    except:
        pass

    if get_dummies:
        df_extended[cat_col] = df_extended[cat_col].astype(str).astype('category')
        df_categorical = pd.get_dummies(df_extended, prefix='ICL-Size', columns=[cat_col])
    else:
        df_categorical = df_extended

    return df_categorical


def predict_ols_for_all_items(array_dict, model, cat_map):
    predictions = {}

    for key, value in array_dict.items():
        X = value.values
        X = sm.add_constant(value, has_constant='add')
        pred = model.predict(X)
        df = pd.DataFrame(data=pred).rename(mapper=cat_map, axis=0).rename(columns={0: key})
        predictions[key] = df

    predictions_df = pd.DataFrame()
    for key, value in predictions.items():
        predictions_df = pd.concat([predictions_df, value], axis=1)

    return predictions_df


def predict_for_all_items(df_dict, model, cat_map):
    predictions = {}

    for key, value in df_dict.items():
        pred = model.predict(value)
        df = pd.DataFrame(data=pred).rename(mapper=cat_map, axis=0).rename(columns={0: key})
        predictions[key] = df

    predictions_df = pd.DataFrame()
    for key, value in predictions.items():
        predictions_df = pd.concat([predictions_df, value], axis=1)

    return predictions_df
