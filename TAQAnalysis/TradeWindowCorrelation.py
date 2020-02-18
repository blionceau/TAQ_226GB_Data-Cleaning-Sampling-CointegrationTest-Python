import numpy as np
from scipy.stats import chi2
from statsmodels.tsa.stattools import adfuller
from TAQImpactUtils.LastPriceBuckets import *
from statsmodels.stats.diagnostic import acorr_ljungbox

class TradeWindowCorrelation(object):
    """
    The class to choose the best window to use the last trade price, such that,
    The price sequence will not suffer from auto-correlation: bid-ask bounce
    """

    def __init__(self, _interdayTrades, _k, _interval):
        """
        To build a class with trade data
        :param _interdayTrades:
        :param _k: the lags for calculating Box test
        :param _interval, also known as window size

        Since we need to use price for each interval (bucket), we choose to use the last bucket
        """
        self._interdayTrades = _interdayTrades
        self._k = _k
        self._interval = _interval * 1000 # in milliseconds
        self._buckets = int(23400)
        self._price = np.zeros(int(self._buckets * len(_interdayTrades.trades.keys())))

        _dates = list(_interdayTrades.trades.keys())
        _dates.sort()
        for i in range(len(_dates)):
            lastprice = LastPriceBuckets(self._interdayTrades.trades[_dates[i]],
                                         23400,
                                         19 * 60 * 60 * 1000 / 2,
                                         16 * 60 * 60 * 1000)
            for j in range(lastprice.getN()):
                self._price[i*self._buckets + j] = lastprice.getPrice(j)
        self._price = self._price[~(self._price == None)]
        self._price = self._price[~np.isnan(self._price)]

    def __repr__(self):
        return "Calculate Trade Window correlation hypothesis test when window size = {}".format(
            self._interval/1000)


    @staticmethod
    def BoxTest(seq, k=None):
        """
        :param k: default 12*(nobs/100)^{1/4}
        :param seq: sequence to be tested
        :return: test value, critical value, test result.
        """
        _seq = np.copy(seq)
        _seq = _seq[~((_seq == None) | np.isnan(_seq))]
        _seq = _seq[:-1]/_seq[1:] - 1
        '''
        box = acorr_ljungbox(seq, k)
        print(box)
        '''
        n = len(_seq)
        if k is None:
            k = int(12 * (n / 100) ** .25)
        autocorr = np.zeros(k)
        denominators = np.zeros(k)
        for i in range(1, k+1):
            autocorr[i-1] = np.corrcoef(_seq[:-i], _seq[i:])[0, 1] ** 2
            denominators[i-1] = n-i
        
        return (n+2)*n*sum(autocorr / denominators), chi2.ppf(0.995, k), \
               'No AutoCorrelation' if (n-2)*n*sum(autocorr / denominators) < chi2.ppf(0.99, k) else 'AutoCorrelation'

        '''
        passes = (box[0] < box[1])
        return '{0} lags passed no autocorr, {1} lags not passed, rej no autocorr'.format(sum(passes), sum(~passes))
        '''

    @staticmethod
    def ADFTest(seq, k=None):
        """
        Doing the ADF test of stationarity
        :param seq: sequence to test
        :param k: the lag to specify
        :return:
        """
        _seq = np.copy(seq)
        _seq = _seq[~((_seq == None) | np.isnan(_seq))]
        _seq = _seq[:-1] / _seq[1:] - 1

        adf = adfuller(_seq, k)
        return adf[0], adf[4]['1%'], 'Stationary' if adf[0] < adf[4]['1%'] else 'Non Stationary'

    def test(self, on='serialcorrelation'):
        print('Running test on {0} observations of trades, window size {1}s'.format(
            self._price.shape[0], self._interval / 1000))
        if on == 'serialcorrelation':
            return TradeWindowCorrelation.BoxTest(self._price[
                                                  int(self._interval/1000)-1::int(self._interval/1000)],
                                                  self._k)
        if on == 'stationarity':
            return TradeWindowCorrelation.ADFTest(self._price[
                                                  int(self._interval/1000)-1::int(self._interval/1000)],
                                                  self._k)



    @property
    def interval(self):
        return self._interval

    @interval.setter
    def interval(self, val):
        self._interval = val * 1000

