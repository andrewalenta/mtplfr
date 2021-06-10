from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.model_selection import KFold, cross_validate

def mutli_model(df):
    """ Function to determine best model archietecture """

    y = []
    for elems in df['Frequency']:
        if elems > 0:
            y.append(1)
        else:
            y.append(0)


    X = df.drop(columns=['Frequency'])  # ['VehPower', 'VehAge', 'DrivAge', 'BonusMalus', 'VehBrand', 'VehGas', 'Area', 'Density', 'Region']
    X1 = df[['DrivAge', 'VehPower', 'VehAge', 'Density', 'BonusMalus']]
    cat_data = ['VehBrand', 'VehGas', 'Area', 'Region']
    num_data = ['VehPower', 'VehAge', 'DrivAge', 'BonusMalus', 'Density']

    dum_Vehbrand = pd.get_dummies(X['VehBrand'])
    dum_VehGas = pd.get_dummies(X['VehGas'])
    dum_Area = pd.get_dummies(X['Area'])
    dum_Region = pd.get_dummies(X['Region'])

    X.append(['dum_Vehbrand', 'dum_VehGas', 'dum_Area', 'dum_Region'])
    X = df.drop(columns=['VehBrand', 'VehGas', 'Area', 'Region'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


    #dfs = []
    models = [
        ('LogReg', LogisticRegression()),
        ('RF', RandomForestClassifier()),
        ('KNN', KNeighborsClassifier()),
        ('SVM', SVC()),
        ('GNB', GaussianNB()),
        ('XGB', XGBClassifier(eval_metric="error"))
    ]

    results = []
    names = []
    scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'roc_auc']
    target_names = ['App_Status_1', 'App_Status_2']

    for name, model in models:
        kfold = KFold(n_splits=5, shuffle=True, random_state=90210)
        cv_results = cross_validate(model, X_train, y_train, cv=kfold, scoring=scoring)
        clf = model.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(name)
        print(classification_report(y_test, y_pred, target_names=target_names))
        results.append(cv_results)
        names.append(name)

        this_df = pd.DataFrame(cv_results)
        this_df['model'] = name
        dfs.append(this_df)

    final = pd.concat(dfs, ignore_index=True)
    return final