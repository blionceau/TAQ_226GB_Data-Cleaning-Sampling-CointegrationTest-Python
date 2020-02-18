import numpy as np


class Trades(object):
    """
    The main class of the trades object. Because the data in TAQQuotesReader is not changable, the implementation of Trades
    object will make the adjustments possible.

    """

    def __init__(self, _ticker, _date, _tradeReader):
        """

        :param _ticker: MSFT, for example
        :param _date: 20070620, str
        :param _tradeReader: TAQTradesReader object
        """

        self._ticker = _ticker
        self._date = _date
        self._tradeReader = _tradeReader
        self._N = _tradeReader.getN()
        self._epoch = _tradeReader.getSecsFromEpocToMidn()
        self._tp = np.zeros(self._N, dtype=float) # trade price
        self._ts = np.zeros(self._N, dtype=int) # trade size
        self._tt = np.zeros(self._N, dtype=int)
        self._dir = None # the trade directions, could be inferred using tick / quote tests

        for i in range(self._N):
            self._tp[i] = self._tradeReader.getPrice(i)
            self._ts[i] = self._tradeReader.getSize(i)
            self._tt[i] = self._tradeReader.getMillisFromMidn(i)

    @property
    def N(self):
        return self._N

    @N.setter
    def N(self, value):
        self._N = value

    @property
    def epoch(self):
        return self._epoch

    @epoch.setter
    def epoch(self, val):
        self._epoch = val

    @property
    def tp(self):
        return self._tp

    @property
    def ts(self):
        return self._ts


    @property
    def tt(self):
        return self._tt

    def __repr__(self):
        return "Trades object of {0} at date {1}".format(self._ticker, self._date)

    def adj(self, cumPriceFactor, cumShareFactor):
        self._tp /= cumPriceFactor # Here we didn't consider the tick size
        # floor the shares to be integer
        self._ts = (self._ts * cumShareFactor).astype(int)

    def setTp(self, idx, val):
        self._tp[idx] = val

    def setTs(self, idx, val):
        self._ts[idx] = val

    def setTt(self, idx, val):
        self._tt[idx] = val

    def clean(self, meanPrice, band):
        idxs = (np.abs(self._tp - meanPrice) <= band)
        self._ts = self._ts[idxs]
        self._tp = self._tp[idxs]
        self._tt = self._tt[idxs]
        self._N = self._ts.shape[0]

    def getN(self):
        return self._N

    def getPrice(self, idx):
        return self._tp[idx]

    def getSize(self, idx):
        return self._ts[idx]

    def getTimestamp(self, idx):
        return self._tt[idx]

    @property
    def dir(self):
        return self._dir

    @dir.setter
    def dir(self, trades_dir):
        self._dir = trades_dir
