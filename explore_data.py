import pandas as pd
import statsmodels.api as sm
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import TweedieRegressor
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
#blabla

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