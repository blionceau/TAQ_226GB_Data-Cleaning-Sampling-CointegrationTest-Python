from math import sqrt
from cvxopt import matrix
from cvxopt.blas import dot
from cvxopt.solvers import qp, options
import pandas as pd
import numpy as np

def meanVarUtils(datname = 'ret_mat_150.npy',
                 pick_path = 'C:\\Users\\lenovo\\Desktop\\Algo Trading\\HW\\HW1\\PlotOutput\\'):

    tickers = np.load(pick_path + 'TIC.npy')

    dat = np.load(pick_path + datname)

    nobs = dat.shape[0]
    info_thr = nobs / 2
    num_nan = sum(np.isnan(dat))
    num_zero = sum(np.isclose(dat, np.zeros(dat.shape)))
    idxs_drop = (num_nan > info_thr) | (num_zero > info_thr)
    print("{} tickers dropped due to loss of information or too few observations".format(idxs_drop.sum()))
    print('Dropped tickers: {}'.format(tickers[idxs_drop]))
    dat_clean = dat[:, ~idxs_drop]
    df_clean = pd.DataFrame(dat_clean, columns=tickers[~idxs_drop])
    return df_clean.cov(), df_clean.mean()


S, pbar = meanVarUtils('ret_mat_300.npy') # first use the 300s sampled data
tic = pbar.index
thr_mkt = np.linalg.inv(np.matrix(S.values)) * np.matrix(pbar.values).T
thr_mkt = np.array(thr_mkt.T)[0]
print(np.array(thr_mkt.T)[0])

n = len(pbar)
S = matrix( S.values )
pbar = matrix( pbar.values ) - 0.02 / (23400 / 150 * 360)

G = matrix(0.0, (n,n))
G[::n+1] = 0.0
h = matrix(0.0, (n,1))
A = matrix(1.0, (1,n))
b = matrix(1.0)

N = 100
mus = [ 10**(5.0*t/N-1.0) for t in range(N) ]
options['show_progress'] = True
xs = [ qp(mu*S, -pbar, G, h, A, b)['x'] for mu in mus ]
returns = [ dot(pbar,x) for x in xs ]#list(map(lambda x: (1+x)**(23400/150*250)-1, [ dot(pbar,x) for x in xs ]))
risks = [ sqrt(dot(x, S*x))  for x in xs ]#* np.sqrt(23400/150*250)

sharpeR = [i/j for i, j in zip(returns, risks)]
max_x = np.argmax(sharpeR)
mkr_portf = xs[max_x]
mkr_portf = pd.Series(mkr_portf, index=tic)

#options['abstol'] scalar (default: 1e-7)
#options['reltol'] scalar (default: 1e-6)

print(returns)

try: import pylab
except ImportError: pass
else:
    pylab.figure(1, facecolor='w')
    pylab.plot(risks, returns)
    pylab.scatter(risks[max_x], returns[max_x])
    pylab.text(risks[max_x], returns[max_x], 'max Sharpe Ratio portfolio')
    pylab.xlabel('standard deviation')
    pylab.ylabel('expected return')
    #pylab.axis([0, 0.2, 0, 0.15])
    pylab.title('Risk-return trade-off curve')
    #pylab.yticks([0.00, 0.05, 0.10, 0.15])
    '''
    pylab.figure(2, facecolor='w')
    c1 = [ x[0] for x in xs ]
    c2 = [ x[0] + x[1] for x in xs ]
    c3 = [ x[0] + x[1] + x[2] for x in xs ]
    c4 = [ x[0] + x[1] + x[2] + x[3] for x in xs ]
    pylab.fill(risks + [.20], c1 + [0.0], facecolor = '#F0F0F0')
    pylab.fill(risks[-1::-1] + risks, c2[-1::-1] + c1,
        facecolor = '#D0D0D0')
    pylab.fill(risks[-1::-1] + risks, c3[-1::-1] + c2,
        facecolor = '#F0F0F0')
    pylab.fill(risks[-1::-1] + risks, c4[-1::-1] + c3,
        facecolor = '#D0D0D0')
    pylab.axis([0.0, 0.2, 0.0, 1.0])
    pylab.xlabel('standard deviation')
    pylab.ylabel('allocation')
    pylab.text(.15,.5,'x1')
    pylab.text(.10,.7,'x2')
    pylab.text(.05,.7,'x3')
    pylab.text(.01,.7,'x4')
    pylab.title('Optimal allocations (fig 4.12)')
    '''

    pylab.figure(2, facecolor='w')
    pylab.scatter(thr_mkt, mkr_portf.values, alpha=.1)
    for i, j in zip(thr_mkt, range(mkr_portf.shape[0])):
        pylab.text(i+1, mkr_portf.iloc[j], tic[j], fontsize=5)
    pylab.title('Comparison between theoretical results and optimization using CVXOPT')
    pylab.xlabel('Theoretical')
    pylab.ylabel('Optimization results')
    '''
    pylab.figure(2)
    mkr_portf_ord = mkr_portf.sort_values(ascending=False).iloc[:10]
    mkr_portf_ord.plot(kind='bar')'''
    pylab.show()