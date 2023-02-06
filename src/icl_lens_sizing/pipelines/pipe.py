from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso


def run_pipeline(X_train, X_test, y_train):

    scaler = StandardScaler()
    lasso = Lasso()

    pipeline = Pipeline([('scaler', scaler), ('lasso', lasso)])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    return y_pred
