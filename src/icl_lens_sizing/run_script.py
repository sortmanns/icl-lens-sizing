from preprocessing.preprocessing import load_and_prepare_data

features = ['ACD', 'ACA nasal', 'ACA temporal', 'AtA', 'ACW',
            'ARtARLR', 'StS', 'StS LR', 'CBID', 'CBID LR', 'mPupil', 'WtW MS-39',
            'WtW IOL Master', 'Sphäre', 'Zylinder', 'Sphärisches Äquivalent']

X_train, y_train, X_validation, y_validation, df = load_and_prepare_data(features=features, validation_size=0)
