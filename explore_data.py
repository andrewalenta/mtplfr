import statsmodels.api as sm
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import TweedieRegressor
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def explore_data(df):
    print(df.columns)
  
    y = df['Frequency']
    X = df.drop(columns=['Frequency'])#['VehPower', 'VehAge', 'DrivAge', 'BonusMalus', 'VehBrand', 'VehGas', 'Area', 'Density', 'Region']
    X1 = df[['DrivAge','VehPower','VehAge','Density','BonusMalus']]
    cat_data = ['VehBrand', 'VehGas', 'Area', 'Region']
    num_data = ['VehPower', 'VehAge', 'DrivAge', 'BonusMalus', 'Density']
    
    log_scale_transformer = make_pipeline(
        FunctionTransformer(func=np.log),
        StandardScaler()
    )

    column_trans = ColumnTransformer(
    [
        ("onehot_categorical", OneHotEncoder(),
            ["VehBrand", "VehPower", "VehGas", "Region", "Area"]),
        ("passthrough_numeric", "passthrough",
            ["BonusMalus"]),
        ("log_scaled_numeric", log_scale_transformer,
            ["Density"]),
    ],
    remainder="drop",
    )
    X = column_trans.fit_transform(df)
    
    print(type(y))
    print(type(X))
    mod = sm.OLS(y, X.astype(float))    # Describe model

    res = mod.fit()       # Fit model

    print(res.summary())