from TAQStats import *
from TAQBasics import *
from TAQAdj import *


class meanCovGen2(object):

    @staticmethod
    def TAQcov(interval):
        taqinfo = TAQInfo("../../s_p500.xlsx")
        ticker = list(taqinfo.ticker.keys())
        ticker = [i for i in ticker if str(i)!='nan']
        np.save(r"C:\Users\lenovo\Desktop\Algo Trading\HW\HW1\PlotOutput\TIC", np.array(ticker))
        companies = list(set(taqinfo.ticker.values()))
        print(len(ticker) - len(companies))
        print(len(ticker))

        ret_mat = np.zeros((int(23400/interval) * 65, len(ticker)))

        adjs = taqinfo.adjustment['MSFT']


        _dates = list(adjs.keys())
        _dates.sort()
        print(_dates)

        #ticker = ['MSFT', 'IBM', 'GOOG']

        for i in range(len(ticker)):
            try:
                adjs = taqinfo.adjustment[ticker[i]]
                print('_____________Begin Parsing Data_______________')
                """
                I have two machines, and the paths need to be changed between machines. Therefore, the test file probably 
                will have different paths
                """
                interdayTrades = InterdayTrades(ticker[i], _dates, 'I:\\R\\trades\\')
                print('_____________Begin Processing Data_______________')
                adjuster = TAQAdjust(_dates, ticker[i], None, interdayTrades,
                                     {k: adjs[k][0] for k in adjs.keys()},
                                     {k: adjs[k][1] for k in adjs.keys()})
                adjuster.adj()
                cleaner = TAQCleaner(_dates, ticker[i], None, interdayTrades, 5, .0005)
                cleaner.clean()
                print('_____________Processing finished_______________')


                tradeStats = TradeStats(interdayTrades, interval)
                tradeStats.genReturns()
                ret_mat[:, i] = tradeStats.ret
                print('{0} successfully calculated, {1} left'.format(ticker[i], len(ticker)-i-1))
            except:
                print('Some problem with '+str(ticker[i]))

        np.save(r"C:\Users\lenovo\Desktop\Algo Trading\HW\HW1\PlotOutput\ret_mat_"+str(interval), ret_mat)

        return None

if __name__ == '__main__':
    meanCovGen2.TAQcov(300)
    meanCovGen2.TAQcov(180)