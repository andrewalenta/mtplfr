import dataclasses
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 50)


def get_data():

    df = pd.read_csv('data/freMTPL2freq.csv')

    X = df.drop(columns=['IDpol'])
    BS_col_list = ['VehPower', 'VehAge', 'DrivAge', 'BonusMalus', 'VehBrand', 'VehGas', 'Area', 'Density', 'Region']
    col_to_sum = ['ClaimNb', 'Exposure']
    col_to_group = [col for col in BS_col_list if col not in col_to_sum]
    g_df = X.groupby(col_to_group, as_index=False).sum()

    for col in g_df:
        print(col)
        print(g_df[col].value_counts())

    write_df = False
    if write_df:
        df.to_parquet('df.parquet.gzip', compression='gzip')


    read_df = False
    if read_df:
        g_df = pd.read_parquet('data/df.parquet.gzip')

    g_df["Frequency"] = g_df["ClaimNb"] / g_df["Exposure"]

    print("Average Frequency = {}"
        .format(np.average(g_df["Frequency"], weights=g_df["Exposure"])))

    print("Fraction of exposure with zero claims = {0:.1%}"
        .format(g_df.loc[g_df["ClaimNb"] == 0, "Exposure"].sum() /
                g_df["Exposure"].sum()))
    

#    fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, figsize=(16, 4))
#    ax0.set_title("Number of claims")
#    _ = g_df["ClaimNb"].hist(bins=30, log=True, ax=ax0)
#    ax1.set_title("Exposure in years")
#    _ = g_df["Exposure"].hist(bins=30, log=True, ax=ax1)
#    ax2.set_title("Frequency (number of claims per year)")
#    _ = g_df["Frequency"].hist(bins=30, log=True, ax=ax2)
#    plt.show()

    return g_df



    #todo
    #packages for ml
    #test all algorithms
    #test metrics
    #test plots

    #class MachineLearning:
        
        #train_test_split
        #X,y ...
        #scores
            #randomizedsearch
    #gridsearch
        #featureimportance
        #metrics
