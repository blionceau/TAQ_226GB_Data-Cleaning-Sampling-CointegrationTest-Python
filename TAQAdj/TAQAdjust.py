
class TAQAdjust(object):
    """
    The class is designed to adjust the daily Price and Volume using the adjusted factor given
    from outside the class.

    This class adjusts trades and quotes at the same time.

    Construction method:
        TAQAdjust(_dates, _quotes, _trades, )
    """

    def __init__(self, _dates, _ticker, _quotes, _trades, _cumPriceAdj, _cumShareAdj):
        """
        :param _dates: iterable(str), 20070620
        :param _ticker: in short, MSFT
        :param _quotes: Trades
        :param _trades: Quotes
        :param _cumPriceAdj: dict(str: float), actual price = price / cumulative price adj factor
        :param _cumShareAdj: dict(str: float), actual share = share * cumulative share adj factor
        """
        self._ticker = _ticker
        self._dates = _dates
        self._quotes = _quotes
        self._trades = _trades
        self._cumPriceAdj = _cumPriceAdj
        self._cumShareAdj = _cumShareAdj

    def __repr__(self):
        return 'TAQ Adjustment Object of {} at {}'.format(self._ticker, self._dates)

    def adj(self):
        # First adjust trades
        for i in self._dates:
            print("Performing adjustment on the trades & quotes of {0} on {1}".format(
                self._ticker, i))
            if abs(self._cumPriceAdj[i] - 1.) < 1e-4:
                print('No need to adjust price on {0}'.format(i), end=', ')
            else:
                print('Need to adjust price on {0}'.format(i), end=', ')
            if abs(self._cumShareAdj[i] - 1.) < 1e-4:
                print('No need to adjust volume on {0}'.format(i))
            else:
                print('Need to adjust volume on {0}'.format(i))

            if self._trades!= None:
                self._trades.trades[i].adj(self._cumPriceAdj[i], self._cumShareAdj[i])

            # Then adjust quotes
            if self._quotes!= None:
                self._quotes.quotes[i].adj(self._cumPriceAdj[i], self._cumShareAdj[i])

    def getTrades(self):
        return self._trades

    def getQuotes(self):
        return self._quotes