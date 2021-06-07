import sklearn.datasets

def lasso(df):
    newsgroups = sklearn.datasets.fetch_20newsgroups_vectorized()


    X, y = newsgroups.data, newsgroups.target

    print(newsgroups)