import sklearn.datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso
def lasso(df):

    #newsgroups = sklearn.datasets.fetch_20newsgroups_vectorized()
    y = df['Frequency']
    X = df.drop(columns=[
    'Frequency'])  # ['VehPower', 'VehAge', 'DrivAge', 'BonusMalus', 'VehBrand', 'VehGas', 'Area', 'Density', 'Region']
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
    ridge = Ridge(alpha=0.5, normalize=True).fit(X_train,
                                                 y_train)
    Lasso(alpha=0.1)
   # X, y = newsgroups.data, newsgroups.target

   # print(newsgroups)