import numpy as np
import struct
import gzip

class TAQCleaner(object):
    """
    Data Cleaner of trades and quotes

    """

    def __init__(self, _dates, _ticker, _quotes, _trades, _k, _gamma):
        """
        :param _dates: iterable(str), 20070620. need to be sorted
        :param _ticker: in short, MSFT
        :param _quotes: Trades
        :param _trades: Quotes
        :param _k: number of days
        :param _gamma: the level of gamma (multiplier to mean price)
        """
        self._ticker = _ticker
        self._dates = _dates
        self._quotes = _quotes
        self._trades = _trades
        self._param = (_k, _gamma)
        self._avg_quote = []
        self._avg_trade = []
        self._std_quote = []
        self._std_trade = []

    def __repr__(self):
        return 'TAQ Cleaner Object of {} at {}'.format(self._ticker, self._dates)

    def calBand(self):
        """
        Calculate the band parameter for trades and midquotes separately
        :return:
        """
        avg_trade = np.zeros(len(self._dates))
        avg_quote = np.zeros(len(self._dates))
        std_trade = np.zeros(len(self._dates))
        std_quote = np.zeros(len(self._dates))

        if self._trades != None:
            for i in range(len(self._dates)):
                sums = sum([sum(self._trades.trades[self._dates[j]].tp) for j in range(max(0, i - self._param[0]), i + 1)])
                sumsqr = sum([sum(self._trades.trades[self._dates[j]].tp**2) for j in range(max(0, i-self._param[0]), i+1)])
                nums = sum([self._trades.trades[self._dates[j]].N for j in range(max(0, i - self._param[0]), i + 1)])
                avg_trade[i] = sums / nums
                std_trade[i] = np.sqrt(sumsqr/nums - avg_trade[i] ** 2)
                #print(sumsqr)
                #print(np.std(self._trades.trades[self._dates[i]].tp))

        if self._quotes != None:
            for i in range(len(self._dates)):
                sums_qa = sum([sum(self._quotes.quotes[self._dates[j]].qa) for j in range(max(0, i - self._param[0]), i + 1)])
                sumsqr_qa = sum([sum(self._quotes.quotes[self._dates[j]].qa**2) for j in range(max(0, i-self._param[0]), i+1)])
                sums_qb = sum([sum(self._quotes.quotes[self._dates[j]].qb) for j in range(max(0, i - self._param[0]), i + 1)])
                sumsqr_qb = sum(
                    [sum(self._quotes.quotes[self._dates[j]].qb ** 2) for j in range(max(0, i - self._param[0]), i + 1)])
                nums = sum([self._quotes.quotes[self._dates[j]].N for j in range(max(0, i - self._param[0]), i + 1)])
                avg_qa = sums_qa / nums
                avg_qb = sums_qb / nums

                #print(np.std(self._quotes.quotes[self._dates[i]].qa))

                #print('std', np.sqrt(sumsqr_qa/nums - np.mean(self._quotes.quotes[self._dates[i]].qa)**2))

                avg_quote[i] = (avg_qa + avg_qb) / 2
                # Used the average mid quote
                std_quote[i] = np.sqrt((sumsqr_qa + sumsqr_qb)/nums/2 - avg_quote[i] ** 2)
                # used the larger one among two: looser band

        self._avg_quote = avg_quote
        self._avg_trade = avg_trade
        self._std_quote = std_quote
        self._std_trade = std_trade


    def clean(self):
        print('Calculating the band from given k and gamma')
        self.calBand()
        print('Done band calculating')
        # First adjust trades

        for i in range(len(self._dates)):
            if self._trades!=None:
                print("Performing cleaning on the trades & quotes of {0} on {1}".format(self._ticker, self._dates[i]))
                print('Before cleaning, trades of {0} on {1} has {2} obs'.format(
                    self._ticker,
                    self._dates[i],
                    self._trades.trades[self._dates[i]].N
                ), end=', ')
                self._trades.trades[self._dates[i]].clean(self._avg_trade[i],
                                                   self._param[1] * self._avg_trade[i] + 2 * self._std_trade[i])
                print('After cleaning, there are {0} obs'.format(self._trades.trades[self._dates[i]].N))

            # Then adjust quotes
            if self._quotes != None:
                print('Before cleaning, quotes of {0} on {1} has {2} obs'.format(
                    self._ticker,
                    self._dates[0],
                    self._quotes.quotes[self._dates[i]].N
                ), end=', ')
                self._quotes.quotes[self._dates[i]].clean(self._avg_quote[i],
                                                   self._param[1] * self._avg_quote[i] + 2 * self._std_quote[i])
                print('After cleaning, there are {0} obs'.format(self._quotes.quotes[self._dates[i]].N))

    def getTrades(self):
        return self._trades

    def getQuotes(self):
        return self._quotes

    def to_csv(self, trade_csv_dir, quote_csv_dir):
        """

        :param csv_dir: ends with "\\", like 'C:\\R\\'
        :return:
        """
        import os
        for date in self._dates:
            if not os.path.exists(trade_csv_dir + date):
                os.mkdir(trade_csv_dir + date)
            else:
                pass
            with open(trade_csv_dir + date + '\\' + self._ticker + '_trades.csv', 'w') as csv:
                csv.write("SecondsFromEpoch,Price,Size\n")
                for i in range(self._trades.trades[date].getN()):
                    csv.write("{0},{1},{2}\n".format(self._trades.trades[date].tt[i] +self._trades.trades[date].epoch,
                                                   self._trades.trades[date].tp[i],
                                                   self._trades.trades[date].ts[i]
                                                   ))
            csv.close()

        for date in self._dates:
            if not os.path.exists(quote_csv_dir + date):
                os.mkdir(quote_csv_dir + date)
            else:
                pass
            with open(quote_csv_dir + date + '\\' + self._ticker + '_quotes.csv', 'w') as csv:
                csv.write("SecondsFromEpoch,BidPrice,AskPrice,BidSize,AskSize\n")
                for i in range(self._quotes.quotes[date].getN()):
                    csv.write("{0},{1},{2},{3},{4}\n".format(
                        self._quotes.quotes[date].qt[i] + self._quotes.quotes[date].epoch,
                        self._quotes.quotes[date].qb[i],
                        self._quotes.quotes[date].qa[i],
                        self._quotes.quotes[date].bs[i],
                        self._quotes.quotes[date].As[i],
                        ))
            csv.close()


    def to_bin(self, trade_bin_dir, quote_bin_dir):
        """
        :param trade_bin_dir: end with "\\"
        :param quote_bin_dir: end with "\\"
        :return: nothing
        """
        import os
        for date in self._dates:
            if not os.path.exists(trade_bin_dir + date):
                os.mkdir(trade_bin_dir + date)

            # print('Writing trades files')
            with gzip.open(trade_bin_dir + date + '\\' + self._ticker + '_trades.binRT', 'wb') as bfile:
                bfile.write(struct.pack('>i', self._trades.trades[date].epoch))
                bfile.write(struct.pack('>i', self._trades.trades[date].N))
                bfile.write(struct.pack('>%di' % self._trades.trades[date].N,
                                        *self._trades.trades[date].tt))
                bfile.write(struct.pack('>%di' % self._trades.trades[date].N,
                            *self._trades.trades[date].ts))
                bfile.write(struct.pack('>%df' % self._trades.trades[date].N,
                            *self._trades.trades[date].tp))
                bfile.close()

        for date in self._dates:
            if not os.path.exists(quote_bin_dir + date):
                os.mkdir(quote_bin_dir + date)
            with gzip.open(quote_bin_dir + date + '\\' + self._ticker + '_quotes.binRQ', 'wb') as bfile:
                bfile.write(struct.pack('>i', self._quotes.quotes[date].epoch))
                bfile.write(struct.pack('>i', self._quotes.quotes[date].N))
                bfile.write(struct.pack('>%di' % self._quotes.quotes[date].N,
                                        *self._quotes.quotes[date].qt))
                bfile.write(struct.pack('>%di' % self._quotes.quotes[date].N,
                                        *self._quotes.quotes[date].bs))
                bfile.write(struct.pack('>%df' % self._quotes.quotes[date].N,
                                        *self._quotes.quotes[date].qb))
                bfile.write(struct.pack('>%di' % self._quotes.quotes[date].N,
                                        *self._quotes.quotes[date].As))
                bfile.write(struct.pack('>%df' % self._quotes.quotes[date].N,
                                        *self._quotes.quotes[date].qa))
                bfile.close()
        print('Files to bin success')

