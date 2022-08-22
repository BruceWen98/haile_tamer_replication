import haile_tamer_simulation as HTS
import haile_tamer_estimation as HTE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.special as sp
from tqdm import tqdm

def run_T_simulations(T, n, lbda, increment, distribution='lognormal'):
    all_values = []
    all_bids = []
    for i in range(T):
        values, bids, irrelevant = HTS.run_1_simulation(n, lbda, increment, distribution)
        all_values.append(values)
        all_bids.append(bids)

    return all_values, all_bids

all_values, all_bids = run_T_simulations(T=100, n=6, lbda=0.1, increment=0.1, distribution='lognormal')


# # Plot Value & Bid Distributions
# count, bins, ignored = plt.hist(sum(all_values,[]), 100, density=True, align='mid')
# X = np.linspace(min(bins), max(bins), 200)
# # pdf = (np.exp(-(np.log(X) - 3)**2 / (2 * 1**2))
# #        / (X * 1 * np.sqrt(2 * np.pi)))
# cdf = 1/2 * ( 1 + sp.erf((np.log(X) - 3) / (np.sqrt(2) * 1)) )
# plt.plot(X, cdf, linewidth=2, color='r')
# plt.axis('tight')
# plt.show()


F_U_array = []
F_L_array = []
X2 = np.linspace(0, 200, 50)
for v in tqdm(X2):
    F_U_v = HTE.F_hat_U(all_bids,HTE.calc_M(all_bids),v,-0.001)
    F_U_array.append(F_U_v)
    F_L_v = HTE.F_hat_L(all_bids,HTE.calc_M(all_bids),v,-2,increment=0.1)
    F_L_array.append(F_L_v)

plt.scatter(X2,F_U_array)
plt.scatter(X2,F_L_array)
cdf = 1/2 * ( 1 + sp.erf((np.log(X2) - 3) / (np.sqrt(2) * 1)) )
plt.plot(X2, cdf, linewidth=2, color='r')
plt.show()

