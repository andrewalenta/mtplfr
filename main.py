import sys
import get_data as gd
import ols_mtpl
import glm
import ols_logit as ols_l
import lasso

if __name__ == '__main__':
    df = gd.get_data()
 #   ols_mtpl.ols_mtpl(df)
 #   glm.glm(df)
    ols_l.ols_logit(df)
    lasso.lasso(df)
    #test

