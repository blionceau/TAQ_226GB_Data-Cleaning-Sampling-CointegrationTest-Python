'''
This file reads pickeled data of return, filter that, and calculate the mean/variance vector and matrix

'''
import numpy as np
import pandas as pd

def meanVarUtils(datname = 'ret_mat_300.npy',
                 pick_path = 'C:\\Users\\lenovo\\Desktop\\Algo Trading\\HW\\HW1\\PlotOutput\\'):

    tickers = np.load(pick_path + 'TIC.npy')

    dat = np.load(pick_path + datname)

    nobs = dat.shape[0]
    info_thr = nobs / 2
    num_nan = sum(np.isnan(dat))
    num_zero = sum(np.isclose(dat, np.zeros(dat.shape)))
    idxs_drop = (num_nan > info_thr) | (num_zero > info_thr)
    print("{} tickers dropped due to loss of information or too few observations".format(idxs_drop.sum()))
    print('Dropped tickers: {}'.format(tickers[idxs_drop]))
    dat_clean = dat[:, ~idxs_drop]
    df_clean = pd.DataFrame(dat_clean, columns=tickers[~idxs_drop])
    return df_clean.cov(), df_clean.mean()

if __name__ == '__main__':
    print(meanVarUtils())
