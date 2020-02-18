from TAQBasics.Trades import *
from dbReaders.TAQTradesReader import *
import pandas as pd
from TAQImpactUtils.TickTest import *


class InterdayTrades(object):
    """
    This object stores trades with several dates
    The method of this object cleans the data, and has further method of plotting the results.
    """

    def __init__(self, _ticker, _dates, path):
        """
        Init of Interday Trades object, contains a list of trades
        :param _ticker: TAQ ticker
        :param _dates: iterable dates object
        :param path: the file path, r"I:\\R\\trades\\"
        """
        self._ticker = _ticker
        self._trades = {}
        for i in _dates:
            self._trades[i] = Trades(_ticker, i, TAQTradesReader(path+i+'\\'+_ticker+'_trades.binRT'))
            print('Reading trades data of {0} on {1}'.format(self._ticker, i))
        self._size = self.getsize()
        self._dates = list(_dates)
        self._dates.sort()

    def __repr__(self):
        return "Trades project of {0} dates for {1}".format(len(self._trades.keys()), self._ticker)

    @property
    def ticker(self):
        return self._ticker

    @property
    def trades(self):
        return self._trades

    def getsize(self):
        siz = 0
        for i in self._trades.keys():
            siz += self._trades[i].N
        return siz

    @property
    def size(self):
        return self._size

    def elem_iter(self):
        """
        Iterator
        :return: yield of elements
        """
        for i in self._dates:
            for j in range(self._trades[i].N):
                yield (i, self._trades[i].tt[j], self._trades[i].tp[j],self._trades[i].ts[j])

    def inferDir(self):
        tickTest = TickTest()
        #startOfDay = 19 * 60 * 60 * 1000 / 2
        #endOfDay = 16 * 60 * 60 * 1000
        for i in (self._dates):
            self._trades[i].dir = np.array(tickTest.classifyAll(self._trades[i]))
            print("{0} on date {1} tagged".format(self.ticker, i))

    def genPrice(self):
        """
        Generate time stamps, iterator.
        :return: yield n
        """
        _dates = list(self._trades.keys())
        _dates.sort()
        for i in _dates:
            for j in range(self._trades[i].N):
                yield self._trades[i].tp[j]


    def genTime(self):
        """
        Generate values
        :return: yield n
        """
        _dates = list(self._trades.keys())
        _dates.sort()
        for i in _dates:
            for j in range(self._trades[i].N):
                _t = self._trades[i].tt[j]
                hour = int(_t / 60 / 60 / 1000)
                minutes = int((_t - hour * 60 * 60 * 1000 ) / 60 / 1000)
                seconds = int((_t - hour * 60 * 60 * 1000 - minutes * 60 * 1000 ) / 1000)

                yield pd.Timestamp(i[:4]+'-'+i[4:6]+'-'+i[6:]+' '+str(hour)+":"+str(minutes)+":"+str(seconds))

    def plot(self, figname, path='../PlotOutput/'):
        """
        plot the prices
        """

        _dates = list(self._trades.keys())
        _dates.sort()
        self._size = self.getsize()

        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 12))
        plt.plot(np.arange(self._size), list(self.genPrice()), linewidth=1.0)

        ticks = []
        for i, j in zip(range(0, self._size), self.genTime()):
            if i % int(self._size / 50) == 0:
                ticks.append(j)
            else:
                pass
        plt.xticks(np.arange(0, self._size, int(self._size / 50)), ticks, rotation=90, fontsize=6)
        plt.title("{0} Trade Prices from {1} to {2}".format(self.ticker, _dates[0], _dates[-1]))
        plt.savefig(path + figname + '.png')
