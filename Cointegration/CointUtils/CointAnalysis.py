from Cointegration.CointUtils import *


class CointAnalysis(object):
    """
    The class for cointegration analysis
    Using the previously define class CalStats and Combination
    """
    def __init__(self, df):
        self._stat_calculator = CalStats(df)
        self._stat_calculator.getGamma()
        self._CointPairs = None

    def getCointPairs(self):
        if self._CointPairs is not None:
            return self._CointPairs
        self._CointPairs = [k for k in self._stat_calculator.coint.keys()
                            if self._stat_calculator.coint[k][1] < self._stat_calculator.coint[k][2]]
        return self._CointPairs

    @property
    def CointPairs(self):
        return self._CointPairs

    @property
    def stat_calculator(self):
        return self._stat_calculator
