import numpy as np


class Quotes(object):
    """
    The main class of the quotes object. Because the data in TAQQuotesReader is not changable, the implementation of Trades
    object will make the adjustments possible.

    """

    def __init__(self, _ticker, _date, _quoteReader):
        """

        :param _ticker: MSFT, for example
        :param _date: 20070620, str
        :param _quoteReader: TAQQuotesReader object
        """

        self._ticker = _ticker
        self._date = _date
        self._quoteReader = _quoteReader
        self._N = _quoteReader.getN()
        self._epoch = _quoteReader.getSecsFromEpocToMidn()
        self._qb = np.zeros(self._N, dtype=float)  # bid price
        self._qa = np.zeros(self._N, dtype=float)  # ask price
        self._bs = np.zeros(self._N, dtype=int)  # bid size
        self._as = np.zeros(self._N, dtype=int)  # ask size
        self._qt = np.zeros(self._N, dtype=int)  # quote time

        for i in range(self._N):
            self._qb[i] = self._quoteReader.getBidPrice(i)
            self._qa[i] = self._quoteReader.getAskPrice(i)
            self._bs[i] = self._quoteReader.getBidSize(i)
            self._as[i] = self._quoteReader.getAskSize(i)
            self._qt[i] = self._quoteReader.getMillisFromMidn(i)

    @property
    def N(self):
        return self._N

    @property
    def epoch(self):
        return self._epoch

    @property
    def qb(self):
        return self._qb

    @property
    def qa(self):
        return self._qa

    @property
    def bs(self):
        return self._bs

    @property
    def As(self):
        return self._as

    @property
    def qt(self):
        return self._qt

    def __repr__(self):
        return "Quotes object of {0} at date {1}".format(self._ticker, self._date)

    def adj(self, cumPriceFactor, cumShareFactor):
        self._qb /= cumPriceFactor
        self._qa /= cumPriceFactor
        # floor the shares to be integer
        self._bs = (self._bs * cumShareFactor).astype(int)
        self._as = (self._as * cumShareFactor).astype(int)

    def setQb(self, idx, val):
        self._qb[idx] = val

    def setQa(self, idx, val):
        self._qa[idx] = val

    def setAs(self, idx, val):
        self._as[idx] = val

    def setBs(self, idx, val):
        self._bs[idx] = val

    def clean(self, midPrice, band):
        #print(band)
        idx1 = (np.abs(self._qb - midPrice) <= band)
        idx2 = (np.abs(self._qa - midPrice) <= band)
        idxs = idx1 & idx2
        #print(idxs[:5])
        self._qt = self._qt[idxs]
        self._qa = self._qa[idxs]
        self._qb = self._qb[idxs]
        self._as = self._as[idxs]
        self._bs = self._bs[idxs]
        self._N = self._qt.shape[0]

    def getN(self):
        return self._N

    def getPrice(self, idx):
        return (self._qa[idx] + self._qb[idx]) / 2.

    def getTimestamp(self, idx):
        return self._qt[idx]