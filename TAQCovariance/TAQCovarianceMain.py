from TAQCovariance import *

class TAQCovarianceMain(object):
    """
    This is the main thread of TAQ Robust Covariance Estimation testings,

    We consider a rolling method to reuse data and analyze data

    The result will be exported for further use (comparison and analysis)
    """

    @staticmethod
    def CovarianceComparison(datapath='./ReturnData/'):
        ret_dat, tic_dat = np.load(datapath + 'Ret2Min.npy'), np.load(datapath + 'TicUsed.npy')

        # if there are too many nans, we should definitely drop them
        tics_to_drop = (np.isnan(ret_dat).sum(axis=0) > ret_dat.shape[0] / 2)
        print('Drop {0} tickers due to lack of information'.format(sum(tics_to_drop)))
        tic_dat = tic_dat[~tics_to_drop]
        ret_dat = ret_dat[:, ~tics_to_drop]
        ret_dat = np.nan_to_num(ret_dat)
        print('Shape of return data', ret_dat.shape)

        # Since here we also have around 500 assets, to maintain the RMT effects, we use 1000 training points,
        # and to maintain a 8:2 training/testing ratio, we use 250 testing points

        # To consider the rolling calculation scheme, we rolling every 100 lines, and we stop till it's impossible
        # to roll forward
        num_rolling = (ret_dat.shape[0] - 1250) // 100
        res = []
        for i in range(num_rolling + 1):
            print('______________________Calculating Using Rolling, the {}-th______________________'.format(i+1))
            perf = PerformanceUtils(ret_dat[int(100*i):1250+int(100*i), :], 1000)
            res.append(perf.risks)

        np.save(datapath + 'risks', np.array(res))
        pass


if __name__ == '__main__':
    #TAQCovarianceMain.CovarianceComparison()
    risks = np.load('./ReturnData/risks.npy')

    means = np.zeros((4, 3))
    stds = np.zeros((4, 3))
    means_comp = np.zeros((4, 3))
    stds_comp = np.zeros((4, 3))
    for methods in range(4):
        for g in range(3):
            means[methods, g] = np.mean(risks[:, methods, g])
            stds[methods, g] = np.std(risks[:, methods, g])
            means_comp[methods, g] = np.mean(risks[:, methods, g]- risks[:, 0, g])
            stds_comp[methods, g] = np.std(risks[:, methods, g]- risks[:, 0, g])
    print(means)
    pd.DataFrame(means).to_csv('./ReturnData/mean_risk.csv')
    print(stds)
    pd.DataFrame(stds).to_csv('./ReturnData/std_risk.csv')
    pd.DataFrame(stds_comp).to_csv('./ReturnData/stdcomp_risk.csv')
    pd.DataFrame(means_comp).to_csv('./ReturnData/meancomp_risk.csv')

    import matplotlib.pyplot as plt
    methods = ['Clipped', 'Shrinkage', 'Bias Corr']
    estimators = ['Min variance', 'Omniscient', 'Uniform']
    plt.figure(figsize=(12, 16))
    for i in range(9):
        plt.subplot(3, 3, i+1)
        g = i % 3
        method = i//3
        plt.scatter(np.arange(len(risks[:, 0, g])), risks[:, 0, g], alpha=.2, label='Empirical')
        plt.scatter(np.arange(len(risks[:, method+1, g])), risks[:, method+1, g], alpha=.2, label=methods[method])
        plt.title('{0} vs Empirical, {1}'.format(methods[method], estimators[g]))
        plt.legend()
    plt.tight_layout()
    plt.savefig('./ReturnData/risks.png')
    plt.show()



