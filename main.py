import sys
import get_data as gd
import explore_data as ed


if __name__ == '__main__':
    df = gd.get_data()
    ed.explore_data(df)
    #test