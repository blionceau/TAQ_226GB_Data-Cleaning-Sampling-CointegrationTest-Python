from Cointegration.CointUtils import *
import matplotlib.pyplot as plt
from scipy.stats import t

class Coint_Main(object):
    """
    The main class to run the analysis of cointegration
    """
    @staticmethod
    def run_single(path='./Data/', outpath='./Results/'):
        dat = pd.read_csv(path + 'cointData.csv', index_col=None, header=None)
        coint = CointAnalysis(dat)
        cointPairs = coint.getCointPairs()
        print(cointPairs)
        print('In total, number of cointegrated pairs: ', len(cointPairs))
        with open(outpath+'Pairs.csv', 'w') as f:
            f.write('First,Second\n')
            for i in cointPairs:
                f.write(i+'\n')
            f.close()

        print('Check t values distributions')
        plt.hist([np.log(t.cdf(coint.stat_calculator.coint[k][1], 9997))
                  for k in coint.stat_calculator.coint.keys()])
        plt.title('Empirical Distribution of p-values of Dickey Fuller test')
        #plt.savefig(outpath + 'dist.png')
        plt.show()
        plt.close()

        # randomly choose 4 pairs
        pairs = np.random.choice(cointPairs, 4)
        plt.figure(figsize=(12, 8))
        for i in range(4):
            plt.subplot(2, 2, i+1)
            seq1, seq2 = map(int, pairs[i].split(','))
            plt.plot(np.arange(10000), dat.iloc[:, seq1], label='Sequence #'+str(seq1+1), alpha=.5)
            plt.plot(np.arange(10000), dat.iloc[:, seq2], label='Sequence #' + str(seq2 + 1), alpha=.5)
            plt.legend()
            plt.xlabel('Time')
            plt.ylabel('Price Level')
            plt.title('Example of Cointegration of the pair {0} and {1}'.format(seq1+1, seq2+1))
        plt.tight_layout()
        plt.savefig(outpath + 'pairs_example.png')
        plt.close()

    @staticmethod
    def run_Coint_Dynamic(path='./Data/', outpath='./Results/'):
        dat = pd.read_csv(path + 'cointData.csv', index_col=None, header=None)
        dat = dat[np.array([0, 1, 2, 61, 106, 181, 3, 174])]
        coint = CointAnalysis(dat.iloc[:2000, :])
        dat_values = dat.values
        # The pairs are (0, 61), (1, 106), (2, 181), (3, 174)
        # randomly choose 4 pairs as illustration
        pairs_file = [open(outpath+'0_61.csv', 'w'), open(outpath+'1_106.csv', 'w'),
                      open(outpath + '2_181.csv', 'w'), open(outpath+'3_174.csv', 'w')]
        pairs = [(0, 3), (1, 4), (2, 5), (6, 7)]
        header='PriceX,PriceY,sumX,sumXsqr,sumY,sumYsqr,sumXY,sumXlag,sumYlag,sumXsqrlag,sumYsqrlag,' \
               'sumX_lagY,sumY_lagX,sumXY_lag,sumX_lagX,sumY_lagY,sumXprec,sumYprec,sumXsqrprec, sumYsqrprec,' \
               'sumXYprec,gamma\n'
        for file in pairs_file:
            file.write(header)
        for i in range(2000, 10000, 10):  # ten days as a step
            for pair, file in zip(pairs, pairs_file):
                file.write(','.join(list(map(str, coint.stat_calculator.getStatistics(pair))))+'\n')
            coint.stat_calculator.update_many(dat_values[i:i+10, :])
        for file in pairs_file:
            file.close()
        pass


if __name__ == '__main__':
    Coint_Main.run_single()
    #Coint_Main.run_Coint_Dynamic()
