import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

def explore_data(df):
    print(df.columns)
  
    y = df['Frequency']
    X = df.drop(columns=['Frequency'])#['VehPower', 'VehAge', 'DrivAge', 'BonusMalus', 'VehBrand', 'VehGas', 'Area', 'Density', 'Region']
    X1 = df[['DrivAge','VehPower','VehAge','Density','BonusMalus']]
    cat_data = ['VehBrand', 'VehGas', 'Area', 'Region']
    num_data = ['VehPower', 'VehAge', 'DrivAge', 'BonusMalus', 'Density']

    dum_Vehbrand = pd.get_dummies(X['VehBrand'])
    dum_VehGas = pd.get_dummies(X['VehGas'])
    dum_Area = pd.get_dummies(X['Area'])
    dum_Region = pd.get_dummies(X['Region'])

    X.append(['dum_Vehbrand', 'dum_VehGas', 'dum_Area', 'dum_Region'])
    X = df.drop(columns=['VehBrand', 'VehGas', 'Area', 'Region'])
    mod = sm.OLS(y, X)    # Describe model

    res = mod.fit()       # Fit model try

    print(res.summary())
    X_ols = X.to_numpy()
    freq_brand = ols("Frequency ~ VehPower",data=X).fit()
    print("")
    print(freq_brand.params)