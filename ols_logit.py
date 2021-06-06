from statsmodels.formula.api import logit
import pandas as pd
import statsmodels.api as sm
#from statsmodels.formula.api import ols
from sklearn.model_selection import train_test_split

def ols_logit(df):

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

    mod = sm.OLS(y_train, X_train)  # Describe model

    res = mod.fit()  # Fit model try
    print(res.summary())

    y_pred = res.predict(X_test)

    print("")


    X_ols = X.to_numpy()

    X["propensity"] = y

    freq_brand = logit("propensity ~ VehPower ", data=X).fit()
