import pandas as pd
import numpy as np
import scipy.stats as stats
from TAQBasics import *
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.optimize import minimize

class TAQImpactAnalysis(object):
    """
    Analysis of market impact using TAQ data
    """

    def __init__(self, outpath='./TAQImpactData/'):
        """
        Reads data from given path
        :param outpath: data store path
        """
        self.daily_val = pd.read_csv(outpath + 'TotalDailyValue.csv', index_col=0)
        self.imb = pd.read_csv(outpath + 'imbalance.csv', index_col=0)
        self.totalVol = pd.read_csv(outpath + 'totalVolume.csv', index_col=0)
        self.VWAP1 = pd.read_csv(outpath + 'VWAP_short.csv', index_col=0)
        self.VWAPall = pd.read_csv(outpath + 'VWAP_long.csv', index_col=0)
        self.lookback = pd.read_csv(outpath + 'LookBackValue.csv', index_col=0)
        self.firstPrice = pd.read_csv(outpath + 'firstPrice.csv', index_col=0)
        self.lastPrice = pd.read_csv(outpath + 'lastPrice.csv', index_col=0)
        self.lookbackstd = pd.read_csv(outpath + 'LookBackStd.csv', index_col=0)

        # Use the intersection of ticker names

        names = set.intersection(*list(map(lambda x: set(x.columns),
                                  [
                           self.daily_val, self.imb, self.totalVol, self.VWAP1, self.VWAPall, self.lookback,
                           self.firstPrice, self.lastPrice, self.lookbackstd
                       ])))
        self.names = np.array(list(names))
        # sort indices
        self.dates = np.array(self.daily_val.index)
        self.daily_val, self.imb, self.totalVol, self.VWAP1, self.VWAPall, self.lookback, \
        self.firstPrice, self.lastPrice, self.lookbackstd = map(lambda x: x[self.names].sort_index(ascending=True),
                                                                [
            self.daily_val, self.imb, self.totalVol, self.VWAP1, self.VWAPall, self.lookback,
            self.firstPrice, self.lastPrice, self.lookbackstd
        ])

        self.calc()

    def __repr__(self):
        return "Analysis of TAQ impact model using regression"

    def calc(self):
        # permanent impact
        self.perm = (self.lastPrice - self.firstPrice) / self.firstPrice
        # This is from Almgren's paper
        self.real = (self.VWAP1 - self.firstPrice) / self.firstPrice
        # realized price impact
        #print(self.perm)
        self._temp = self.real - self.perm / 2
        # save the Y and X variable for later
        self._X = (self.imb * 6.5 / 6. / self.lookback)
        self._Y =  (self._temp / self.lookbackstd)

    @staticmethod
    def curve_fit(_Y, _X):
        """Curve fitting using scipy"""
        from scipy.optimize import curve_fit
        func = lambda xdat, beta, eta : eta * xdat ** beta
        popt, pcov = curve_fit(func, _X, _Y)#, bounds=[0.01, [1., 1.0]])
        return popt

    @staticmethod
    def OLS(_Y, _X, add_const=True):
        """
        OLS regression, regress _Y on _X
        :return:
        """
        import statsmodels.api as sm
        if add_const:
            XX = sm.add_constant(_X)
        else:
            XX = _X
        ols = sm.OLS(_Y, XX).fit()
        return ols

    @staticmethod
    def WLS(_Y, _X, add_const=True):
        """
        Reference of method: http://www3.grips.ac.jp/~yamanota/Lecture_Note_10_GLS_WLS_FGLS.pdf
        Used FGLS
        :param _Y: y
        :param _X: x
        :param add_const: constant or not
        :return: a WLS object
        """
        import statsmodels.api as sm
        if add_const:
            XX = sm.add_constant(_X)
        else:
            XX = _X
        ols = sm.OLS(_Y, XX).fit()
        resid = ols.resid
        log_resid_sqr = np.log(resid ** 2)
        ols2 = sm.OLS(log_resid_sqr, XX).fit()
        ws = np.exp(ols2.fittedvalues)
        wls = sm.WLS(_Y, XX, weights=1/ws).fit()
        return wls

    @staticmethod
    def eliminateOutliers(_Y, _X):
        """Delete outliers larger than 99 or smaller than 1 percentile, for both Y and X"""
        bands_high = np.percentile(_X, 99)
        bands_low = np.percentile(_X, 1)
        bounds_high = np.percentile(_Y, 99)
        bounds_low = np.percentile(_Y, 1)
        idxs = (_X <= bands_high) & (_X >= bands_low) & (_Y <= bounds_high) & (_Y >= bounds_low)
        return _Y[idxs], _X[idxs]

    @staticmethod
    def negateObservations(_Y, _X):
        """
        Change the wrong observations by negate them.
        More discussion in my report (Yucheng)

        """
        _Y, _X = np.copy(_Y), np.copy(_X)
        idxs = (_Y <= 0) & (_X <= 0)
        # print(sum(idxs))
        _Y[idxs] = - _Y[idxs]
        _X[idxs] = - _X[idxs] + .00001
        idxs = (_X <= 0) & (_Y >= 0)
        _Y[idxs] *= -1.
        _X[idxs] = - _X[idxs] + .00001
        return _Y, _X

    @staticmethod
    def leastSquared(_Y, _X, preprocessing=True, figid='OLS', path='./TAQImpactOutput/'):
        """
        Perform the least squared error fitting using scipy optimization tools
        :param _Y: Y variable
        :param _X: X variable
        :param preprocessing: whether to preprocess or not
        :param figid: figure - id
        :param path: the output path
        :return: a OptRes object to store the results
        """
        from statsmodels.stats.diagnostic import het_white

        # preprocessing
        if preprocessing:
            _Y, _X = TAQImpactAnalysis.negateObservations(*TAQImpactAnalysis.eliminateOutliers(_Y, _X))
        func = lambda params: sum((_Y - params[0] * (_X ** params[1])) ** 2)

        # Output the data in scatterplots
        plt.figure()
        plt.scatter(_X, _Y, alpha=0.5)
        plt.title(r'Scatter plots of $\frac{h}{\sigma}$-$\frac{X}{VT}$')
        plt.savefig(path + figid + '_scatter_xy' + '.png')
        plt.close()

        #do optimization

        res = minimize(func, np.array([0.1, 0.6]))
        eta, beta = res.x
        resids = (_Y - eta * (_X ** beta))

        # Output the residuals distribution
        plt.figure()
        plt.hist(resids)
        plt.title('Distribution of residuals')
        plt.savefig(path+figid + '_resid_dist'+'.png')
        plt.close()

        # Check heteroscedasticity

        print('No het with p value {0}'.format(het_white(resids, sm.add_constant(_X))[3])
              if het_white(resids, sm.add_constant(_X))[3] < .01
              else 'Het with p value {0}'.format(het_white(resids, sm.add_constant(_X))[3]))

        _x_high = max(_X)

        # Plot fitted curve

        plt.figure()
        plt.scatter(_X, _Y, alpha=.3)
        plt.plot(np.arange(0, _x_high, 0.0001), eta * (np.arange(0, _x_high, 0.0001)) ** beta, c='orange')
        plt.xlabel(r'$\frac{X}{VT}$')
        plt.ylabel(r'$\frac{h}{\sigma}$')
        plt.legend(['Fitted curve'])
        plt.title('Fitted Curve of non-linear regression')
        plt.savefig(path+figid + '_curve_fit'+'.png')
        plt.close()

        # calculate T-values

        tvalues = TAQImpactAnalysis.calTStats(_Y, _X, resids)
        return TAQImpactAnalysis.OptRes((eta, beta), [i/j for i, j in zip((eta, beta), tvalues)])


    @staticmethod
    def calTStats(_Y, _X, resids):
        """
        Calculate the SE fr the calculation of T-stats.

        This method is from Linear Regression T-stats calculation

        :param _Y: Y
        :param _X: X
        :param resids: residuals
        :return: a group of t-stats
        """
        sigma2 = np.var(resids)
        n = len(resids)
        xmean = np.mean(_X)
        xvar = np.var(_X)
        t1 = np.sqrt(sigma2 * (1/n + xmean**2 / xvar))
        t2 = np.sqrt(np.var(_Y) / xvar / (n-2))
        return t1, t2


    @staticmethod
    def weightedLeastSquared(_Y, _X, figid='WLS', path='./TAQImpactOutput/'):
        """
        Conduct least square using weights estimated from errors.
        :param _Y:
        :param _X:
        :param figid:
        :param path: all same as LeastSquared function
        :return:
        """

        # Run first fittings

        res_min1 = TAQImpactAnalysis.leastSquared(_Y, _X)
        _Y, _X = TAQImpactAnalysis.negateObservations(*TAQImpactAnalysis.eliminateOutliers(_Y, _X))
        # func = lambda params: sum((_Y - params[0] * (_X ** params[1])) ** 2)

        # Determine weights from the results from first fitting, using method similar to WLS regression

        eta, beta = res_min1.params
        eta_, beta_ = eta, beta
        resids = (_Y - eta * (_X ** beta))
        resids_ = resids
        # regressing log(u^2) ~ X, find more discussion within my report (Yucheng)
        log_sqr_resid = np.log(resids ** 2)
        w_inv = np.exp(-sm.OLS(log_sqr_resid, sm.add_constant(_X)).fit().fittedvalues / 2.)

        func_wls = lambda params: sum((_Y * w_inv - params[0] * (_X ** params[1]) * w_inv) ** 2)
        res = minimize(func_wls, np.array([0.1, 0.6]))
        eta, beta = res.x
        resids = (_Y - eta * (_X ** beta))

        # Plot fitted outputs

        _x_high = np.max(_X)

        plt.figure()
        plt.scatter(_X, _Y, alpha=.3)
        plt.plot(np.arange(0, _x_high, 0.0001), eta * (np.arange(0, _x_high, 0.0001)) ** beta, c='orange')
        plt.plot(np.arange(0, _x_high, 0.0001), eta_ * (np.arange(0, _x_high, 0.0001)) ** beta_, c='brown')
        plt.xlabel(r'$\frac{X}{VT}$')
        plt.ylabel(r'$\frac{h}{\sigma}$')
        plt.legend(['Fitted curve (W-LS)', 'Fitted Curve (O-LS)'])
        plt.title('Fitted Curve of non-linear regression (Weighted Least Squared)')
        plt.savefig(path + figid + '_curve_fit' + '.png')
        plt.close()

        # Compare results: histogram

        plt.figure()
        plt.hist(resids_, alpha=.5)
        plt.hist(resids, alpha=.2)
        plt.legend(['OLS Residual', 'WLS Residual'])
        plt.title('Comparison between OLS residual and WLS residual')
        plt.savefig(path + figid + '_resid_comp' + '.png')
        plt.close()

        # Compare results: error plots

        plt.figure()
        plt.scatter(_X, resids_, alpha=.2)
        plt.scatter(_X, resids, alpha=.2)
        plt.legend(['OLS Residual', 'WLS Residual'])
        plt.title('Comparison between OLS residual and WLS residual')
        plt.savefig(path + figid + '_resid_scatter' + '.png')
        plt.close()

        # error qq plots
        plt.figure()
        sm.qqplot(resids, line='s')
        plt.title('Residuals of OLS')
        plt.savefig(path + figid + '_qqplot_resid_OLS' + '.png')

        plt.figure()
        sm.qqplot(resids_, line='s')
        plt.title('Residuals of WLS')
        plt.savefig(path + figid + '_qqplot_resid_WLS' + '.png')

        tvalues = TAQImpactAnalysis.calTStats(_Y, _X, resids)
        return TAQImpactAnalysis.OptRes((eta, beta), [i/j for i, j in zip((eta, beta), tvalues)])

    class OptRes(object):
        """
        A class to store the results from optimization
        params: the parameters
        tvalues: the T-values associated with params
        """

        def __init__(self, params, tvalues):
            self._params = params
            self._tvalues = tvalues

        @property
        def params(self):
            return self._params

        @property
        def tvalues(self):
            return self._tvalues

        def __repr__(self):
            return "params: {0}, tvalues: {1}".format(self.params, self.tvalues)


    def reg(self, on='all', regtype='OLS', names=None):
        """
        Do regression of everything
        :param on: the names to use, either all, or upper, lower (most/least liquid half)
        :param regtype: OLS or WLS
        :param names: if names != all, you need to specify the names to be used for regression
        :return: OptRes with parameters and their t-values stored
        """
        # import statsmodels.api as sm


        if on == 'all':
            # Put all data together as a vector
            XX = self._X.values.reshape((1, self._X.shape[0]*self._X.shape[1], 1))[0]
            YY = self._Y.values.reshape((1, self._X.shape[0]*self._X.shape[1], 1))[0]

            if regtype == 'OLS':
                return TAQImpactAnalysis.leastSquared(YY, XX, figid='OLS_all')
            else:
                return TAQImpactAnalysis.weightedLeastSquared(YY, XX, figid='WLS_all')
        else:
            # Choose the names to be used, and put everythin into a vector
            XX = self._X[names].values#.reshape((1, self._X.shape[0]*self._X.shape[1], 1))[0]
            XX = (XX.reshape((1, XX.shape[0] * XX.shape[1])))[0]
            YY = self._Y[names].values
            YY = (YY.reshape((1, YY.shape[0] * YY.shape[1])))[0]

            if regtype == 'OLS':
                return TAQImpactAnalysis.leastSquared(YY, XX, figid='OLS_'+on)
            else:
                return TAQImpactAnalysis.weightedLeastSquared(YY, XX, figid='WLS_'+on)


    @staticmethod
    def to_txt(reg_res, txtname, txtpath='./TAQImpactOutput/'):
        """
        Output everything into txt
        """
        with open(txtpath + txtname + '.txt', 'w') as f:
            f.write('eta = {0}\n'.format(reg_res.params[0]))
            f.write('t-eta = {0}\n'.format(reg_res.tvalues[0]))
            f.write('beta = {0}\n'.format(reg_res.params[1]))
            f.write('t-beta = {0}\n'.format(reg_res.tvalues[1]))
            f.close()
        print('Result logged to {0}{1}.txt'.format(txtpath, txtname))

    def getMostLiquidNames(self):
        """
        Get the most liquid half names
        """
        taqliquid = TAQLiquid()
        taqliquid.update(self.names)
        return taqliquid.getLargestHalf()

    def getLeastLiquidNames(self):
        """
        Get the most illuqid half names
        """
        taqliquid = TAQLiquid()
        taqliquid.update(self.names)
        return taqliquid.getSmallestHalf()

    def analysis(self, outpath='./TAQImpactOutput/'):
        """
        Perform analysis for all, most liquid half and least liquid half stocks
        Using OLS and GLS. Exporting result to txt files.
        :return: None
        """
        #XX = self._X.values.reshape((1, self._X.shape[0] * self._X.shape[1], 1))[0]
        #YY = self._Y.values.reshape((1, self._X.shape[0] * self._X.shape[1], 1))[0]
        #print(TAQImpactAnalysis.leastSquared(YY, XX))
        names = None
        for on in ['all', 'upper', 'lower']:
            if on == 'upper':
                names = self.getMostLiquidNames()
            elif on == 'lower':
                names = self.getLeastLiquidNames()
            for regtype in ['OLS', 'WLS']:
                regres = self.reg(on, regtype, names)
                print(on, regtype, '__________________________________________________________')
                TAQImpactAnalysis.to_txt(regres, 'params_part1_{0}_{1}'.format(on, regtype), outpath)

    @property
    def temp(self):
        return self._temp

if __name__ == '__main__':
    # Run these two lines of code compeletes the non-linear regression with regression analysis
    analysis = TAQImpactAnalysis()
    analysis.analysis()