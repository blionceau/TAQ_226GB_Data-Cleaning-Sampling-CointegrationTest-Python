from TAQBasics.Quotes import *
from dbReaders.TAQQuotesReader import *
import pandas as pd

class InterdayQuotes(object):
    """
    This object stores Quotes with several dates
    The method of this object cleans the data, and has further method of plotting the results.
    """

    def __init__(self, _ticker, _dates, path):
        """
        Init of Interday Quotes object, contains a list of trades
        :param _ticker: TAQ ticker
        :param _dates: iterable dates object
        :param path: the file path, r"I:\\R\\trades\\"
        """
        self._ticker = _ticker
        self._quotes = {}
        for i in _dates:
            self._quotes[i] = Quotes(_ticker, i, TAQQuotesReader(path + i + '\\' + _ticker + '_quotes.binRQ'))
            print('Reading quotes data of {0} on {1}'.format(self._ticker, i))
        self._size = self.getsize()

    def __repr__(self):
        return "Quotes project of {0} dates for {1}".format(len(self._quotes.keys()), self._ticker)

    @property
    def ticker(self):
        return self._ticker

    @property
    def quotes(self):
        return self._quotes

    def getsize(self):
        siz = 0
        for i in self._quotes.keys():
            siz += self._quotes[i].N
        return siz

    @property
    def size(self):
        return self._size

    def elem_iter(self):
        """
         Iterator
         :return: yield of elements
         """
        _dates = list(self._quotes.keys())
        _dates.sort()
        for i in _dates:
            for j in range(self._quotes[i].N):
                yield (i, self._quotes[i].qt[j],
                       self._quotes[i].qa[j], self._quotes[i].qb[j],
                       self._quotes[i].As[j], self._quotes[i].bs[j]
                       )

    def genPrice(self):
        """

        :return: yield n
        """
        _dates = list(self._quotes.keys())
        _dates.sort()
        for i in _dates:
            for j in range(self._quotes[i].N):
                yield (self._quotes[i].qa[j] + self._quotes[i].qb[j]) / 2


    def genTime(self):
        """

        :return: yield n
        """
        _dates = list(self._quotes.keys())
        _dates.sort()
        for i in _dates:
            for j in range(self._quotes[i].N):
                _t = self._quotes[i].qt[j]
                hour = int(_t / 60 / 60 / 1000)
                minutes = int((_t - hour * 60 * 60 * 1000 ) / 60 / 1000)
                seconds = int((_t - hour * 60 * 60 * 1000 - minutes * 60 * 1000 ) / 1000)

                yield pd.Timestamp(i[:4]+'-'+i[4:6]+'-'+i[6:]+' '+str(hour)+":"+str(minutes)+":"+str(seconds))

    def plot(self, figname, path='C:\\Users\\yw338\\Desktop\\Algo Trading\\HW\\HW1\\PlotOutput\\'):
        """
        plot the prices
        """

        _dates = list(self._quotes.keys())
        _dates.sort()
        self._size = self.getsize()

        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 12))
        plt.plot(list(range(self._size)), list(self.genPrice()), linewidth=1.0)

        ticks = []
        for i, j in zip(range(0, self._size), self.genTime()):
            if i % int(self._size / 50) == 0:
                ticks.append(j)
            else:
                pass
        plt.xticks(np.arange(0, self._size, int(self._size / 50)), ticks, rotation=90, fontsize=6)
        plt.title("{0} Quote Prices from {1} to {2}".format(self.ticker, _dates[0], _dates[-1]))
        plt.savefig(path + figname + '.png')

