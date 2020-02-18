import pandas as pd
import numpy as np
from Cointegration.CointUtils.Combination import *
from collections import deque
from scipy.stats import t


class CalStats(object):
    """
    Calculate the statistics of price sequence for the use of cointegration analysis
    """
    def __init__(self, df):
        """
        Initialization of the statistics object
        :param df: a dataframe, with rows from different time, and columns from different assets
        """
        self._df = df.values
        self._ind, self._col = df.index, df.columns
        self._T, self._N = self._df.shape
        self._comb = Combination(self._N)
        self._cross_stats = self._comb.getPairsDict()
        self._coint = self._comb.getPairsDict()
        self._stats = self._comb.getSingleDict()
        self._t_critical = t.ppf(1e-25, self._T - 3)
        self._flag = False

    def cal(self):
        """
        Calculate the statistics of a given data frame
        :return: None
        """

        # First we calculate the unary statistics
        # The order in the list is:
        # sum(x_i), sum(x_i**2), sum(x_{i-1}),
        # sum(x_{i-1}**2), sum(x_i x_{i-1}), sum(x_{i+1}), sum(x_{i+1}**2)
        for i in range(self._N):
            if self._stats[i] is None:
                self._stats[i] = [self._df[:, i].sum(), (self._df[:, i]**2).sum()]
                self._stats[i].append(self._stats[i][0] - self._df[-1, i])
                self._stats[i].append(self._stats[i][1] - self._df[-1, i] ** 2)
                self._stats[i].append(np.dot(self._df[1:, i], self._df[:-1, i]))
                self._stats[i].append(self._stats[i][0] - self._df[0, i])
                self._stats[i].append(self._stats[i][1] - self._df[0, i] ** 2)

        # Then we want to calculate binary statistics
        # The order in the lst is:
        # sum(x_i y_i), sum(x_i y_{i-1}),
        # sum(x_{i-1} y_i), sum(x_{i-1} y_{i-1}), sum(x_{i+1}, y_{i+1})
        for k in self._cross_stats.keys():
            _ii, _jj = map(int, k.split(','))
            self._cross_stats[k] = [np.dot(self._df[:, _ii], self._df[:, _jj])]
            self._cross_stats[k].append(np.dot(self._df[1:, _ii], self._df[:-1, _jj]))
            self._cross_stats[k].append(np.dot(self._df[1:, _jj], self._df[:-1, _ii]))
            self._cross_stats[k].append(self._cross_stats[k][0] - self._df[-1, _ii]*self._df[-1, _jj])
            self._cross_stats[k].append(self._cross_stats[k][0] - self._df[0, _ii] * self._df[0, _jj])

        self._flag = True

    def update(self, obs):
        """
        Update the dataframe when a new observation is available
        :return: Nothing
        """
        if not self._flag:
            self.cal()

        # For updating analysis, we need to change the data structures to deque
        if type(self._df) == np.ndarray:
            self._df = [deque(self._df[:, i]) for i in range(self._N)]
        else:
            # It's already list of deques
            pass

        # The total time length is unchanged
        # Get the old data and add new data in storage
        _old_data = [dq.popleft() for dq in self._df]

        # Update stats
        # The order in the list is:
        # sum(x_i), sum(x_i**2), sum(x_{i-1}),
        # sum(x_{i-1}**2), sum(x_i x_{i-1}), sum(x_{i+1})
        # sum(x_{i+1}**2)
        for i in range(self._N):
            self._stats[i][0] += obs[i]
            self._stats[i][0] -= _old_data[i]
            self._stats[i][1] += obs[i] ** 2
            self._stats[i][1] -= _old_data[i] ** 2
            self._stats[i][2] += (self._df[i][-1] - _old_data[i])
            self._stats[i][3] += (self._df[i][-1] ** 2 - _old_data[i] ** 2)
            self._stats[i][4] += (self._df[i][-1] * obs[i] - self._df[i][0] * _old_data[i])
            self._stats[i][5] += (obs[i] - self._df[i][0])
            self._stats[i][6] += (obs[i] ** 2 - self._df[i][0] ** 2)


        # Update Cross Stats
        # The order in the lst is:
        # sum(x_i y_i), sum(x_i y_{i-1}),
        # sum(x_{i-1} y_i), sum(x_{i-1} y_{i-1}), sum(x_{i+1} y_{i+1})
        for k in self._cross_stats.keys():
            _ii, _jj = map(int, k.split(','))
            self._cross_stats[k][0] += obs[_ii] * obs[_jj]
            self._cross_stats[k][0] -= _old_data[_ii] * _old_data[_jj]
            self._cross_stats[k][1] += (obs[_ii] * self._df[_jj][-1] - _old_data[_jj] * self._df[_ii][0])
            self._cross_stats[k][2] += (obs[_jj] * self._df[_ii][-1] - _old_data[_ii] * self._df[_jj][0])
            self._cross_stats[k][3] += (self._df[_jj][-1] * self._df[_ii][-1] - _old_data[_ii] * _old_data[_jj])
            self._cross_stats[k][4] += (obs[_ii] * obs[_ii] - self._df[_jj][0] * self._df[_ii][0])

        # Append new data into dequeues
        for i in range(self._N):
            self._df[i].append(obs[i])

    def update_many(self, obs):
        """
        Updating many observations at a time
        :param obs: each row is a list of observations at a time
        :return: nothing
        """
        for i in range(obs.shape[0]):
            self.update(obs[i, :])

    @staticmethod
    def calGamma(sumX, sumXsqr,
                 sumY, sumYsqr,
                 sumXY,
                 sum_X_lag, sum_Y_lag,
                 sum_x_sqr_lag, sum_y_sqr_lag,
                 sumX_lagY, sumY_lagX,
                 sumXY_lag,
                 sumX_lagX, sumY_lagY,
                 sumX_prec, sumY_prec,
                 sumXsqr_prec, sumYsqr_prec,
                 sumXY_prec, N, t_critical):
        m = (sumXY / N - sumX / N * sumY / N) / (sumXsqr / N - sumX ** 2 / N / N)
        b = (sumY - m * sumX) / N
        _uu = sumY_lagY - m * sumY_lagX - b * sumY_prec - m * sumX_lagY + m ** 2 * sumX_lagX \
            + b * m * sumX_prec - b * sum_Y_lag + b * m * sum_X_lag + b ** 2 * (N - 1)
        _dd = sum_y_sqr_lag + m ** 2 * sum_x_sqr_lag + b ** 2 * (N-1) + 2 * m * b * sum_X_lag \
            - 2 * m * sumXY_lag - 2 * b * sum_Y_lag
        _gamma = _uu / _dd

        # Standard error
        _uu_t = sumYsqr_prec + m ** 2 * sumXsqr_prec + b ** 2 * (N - 1) - 2*m*sumXY_prec - 2*b*sumY_prec \
                + 2*b*m*sumX_prec \
            + _gamma ** 2 * (sum_y_sqr_lag + m ** 2 * sum_x_sqr_lag + b ** 2 * (N - 1)
                             - 2*m*sumXY_lag - 2*b*sum_Y_lag + 2*b*m*sum_X_lag)\
            - 2*_gamma * (sumY_lagY -m*sumY_lagX-b*sumY_prec-m*sumX_lagY + m**2*sumX_lagX + m*b*sumX_prec
                          -b*sum_Y_lag + b*m*sum_X_lag + b ** 2 * (N-1))
        stde = np.sqrt(_uu_t / _dd / (N - 2 - 1))
        _gamma_t = (_gamma - 1) / stde
        return _gamma, _gamma_t, t_critical

    def getGamma(self):
        if not self._flag:
            self.cal()
        # Calculate Gamma for each pair
        for k in self._coint.keys():
            _ii, _jj = map(int, k.split(','))
            # The first is treated as X and second Y
            statsX, statsY, statsXY = self._stats[_ii], self._stats[_jj], self._cross_stats[k]

            _gamma = CalStats.calGamma(statsX[0], statsX[1], statsY[0], statsY[1],
                                       statsXY[0], statsX[2], statsY[2], statsX[3],
                                       statsY[3], statsXY[1], statsXY[2], statsXY[3],
                                       statsX[4], statsY[4], statsX[5], statsY[5],
                                       statsX[6], statsY[6], statsXY[4],
                                       self._T, self._t_critical)
            self._coint[k] = _gamma

    @property
    def stats(self):
        return self._stats

    @property
    def cross_stats(self):
        return self._cross_stats

    @property
    def coint(self):
        return self._coint

    def getStatistics(self, pair):
        _ii, _jj = pair
        k = str(min(_ii, _jj))+','+str(max(_ii, _jj))
        statsX, statsY, statsXY = self._stats[_ii], self._stats[_jj], self._cross_stats[k]
        return (self._df[_ii][-1], self._df[_jj][-1],
                statsX[0], statsX[1], statsY[0], statsY[1],
               statsXY[0], statsX[2], statsY[2], statsX[3],
               statsY[3], statsXY[1], statsXY[2], statsXY[3],
               statsX[4], statsY[4], statsX[5], statsY[5],
               statsX[6], statsY[6], statsXY[4], self._coint[k][0])