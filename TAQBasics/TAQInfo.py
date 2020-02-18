import pandas as pd
import numpy as np

class TAQInfo(object):
    """
    This class contains the information of TAQ data, read from s_p500.xlsx

    The contents of TAQInfo:
    - TAQ stock name <-> ticker name
    - TAQ dates
    - TAQ Adjustment factors

    """

    def __init__(self, snpfilepath):
        """
        parse the data from given snp500.xlsx file
        :param snpfilepath: the corresponding snpfilepath
        """
        _df = pd.read_excel(snpfilepath) # read dataFrame
        _df = _df[_df['Names Date'].notnull()] # filter out the null data
        # useful columns
        _names = np.array(['Ticker Symbol', 'Company Name', 'Names Date', 'Cumulative Factor to Adjust Prices',
                          'Cumulative Factor to Adjust Shares/Vol'])
        _df = _df[_names]
        self._ticker = {} # Use ticker to find company names
        self._dates = set(map(lambda x: str(int(x)), _df['Names Date'].values)) # A list of trading dates
        self._adjustments = {} # A dictionary of dictionary. ticker -> dates -> (adjPrice, adjShare)
        for i in range(_df.shape[0]):
            _tick, _name, _date, _adjP, _adjS = _df.iloc[i, :]
            _date = str(int(_date))
            if not self._ticker.get(_tick):
                self._ticker[_tick] = _name
            if not self._adjustments.get(_tick):
                self._adjustments[_tick] = {_date: (_adjP, _adjS)}
            else:
                self._adjustments[_tick][_date] = (_adjP, _adjS)

    @property
    def ticker(self):
        return self._ticker

    @property
    def dates(self):
        return self._dates

    @property
    def adjustment(self):
        return self._adjustments

    @ticker.setter
    def ticker(self, _ticker):
        self.ticker = _ticker

    @dates.setter
    def dates(self, _dates):
        self._dates = _dates

    @adjustment.setter
    def adjustment(self, adj):
        self._adjustments = adj

    def __repr__(self):
        return "TAQ Information object."
