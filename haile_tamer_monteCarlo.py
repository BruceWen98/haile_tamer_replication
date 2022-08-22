import haile_tamer_simulation as HTS
import haile_tamer_estimation as HTE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.special as sp

def run_T_simulations(T, n, lbda, increment, distribution='lognormal'):
    all_values = []
    all_bids = []
    for i in range(T):
        values, bids, irrelevant = HTS.run_1_simulation(n, lbda, increment, distribution)
        all_values.append(values)
        all_bids.append(bids)

    return all_values, all_bids

all_values, all_bids = run_T_simulations(T=100, n=6, lbda=0.1, increment=0.1, distribution='lognormal')


# Plot Value & Bid Distributions
count, bins, ignored = plt.hist(sum(all_bids,[]), 100, density=True, align='mid')
X = np.linspace(min(bins), max(bins), 10000)
# pdf = (np.exp(-(np.log(X) - 3)**2 / (2 * 1**2))
#        / (X * 1 * np.sqrt(2 * np.pi)))
# cdf = 1/2 * ( 1 + sp.erf((np.log(X) - 3) / (np.sqrt(2) * 1)) )
# plt.plot(X, cdf, linewidth=2, color='r')
# plt.axis('tight')
# plt.show()



F_U_array = []

for v in X:
    testing = HTE.F_hat_U(all_bids,HTE.calc_M(all_bids),v,-10.0)
    print(testing)

# print(F_U_array)

