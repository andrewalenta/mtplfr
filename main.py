import sys
import get_data as gd
import ols_mtpl


if __name__ == '__main__':
    df = gd.get_data()
    ols_mtpl.ols_mtpl(df)
    #test