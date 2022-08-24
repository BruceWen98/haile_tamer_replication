import haile_tamer_simulation as HTS
import haile_tamer_estimation as HTE
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
from tqdm import tqdm
import collections

def run_T_simulations(T, n, lbda, increment, distribution='lognormal'):
    all_values = []
    all_bids = []
    max_bid_dicts = []
    for i in range(T):
        values, bids, losing_player, max_bid_dict = HTS.run_1_simulation(n, lbda, increment, distribution)
        all_values.append(values)
        all_bids.append(bids)
        
        max_bid_dict = dict(collections.OrderedDict(sorted(max_bid_dict.items())))  # Sort the dict by key, ascending order.
        max_bid_dicts.append(max_bid_dict)

    return all_values, all_bids, max_bid_dicts

all_values, all_bids, max_bid_dicts = run_T_simulations(T=25, n=2, lbda=0.1, increment=1, distribution='lognormal')

print(max_bid_dicts)


# What if we only took the top 2 bids? Even though there are >2 bidders in each auction.
def top2(d):
    top2keys = sorted(d, key=d.get, reverse=True)[:2]
    out_dict = {}
    out_dict[top2keys[0]] = d[top2keys[0]]
    out_dict[top2keys[1]] = d[top2keys[1]]
    return out_dict

max_bid_dicts_top2only = [top2(max_bid_dict) for max_bid_dict in max_bid_dicts]


# # Plot Value & Bid Distributions
# max_bids = []
# for max_bid_dict in max_bid_dicts:
#     max_bids.extend(list(max_bid_dict.values()))

# count, bins, ignored = plt.hist(max_bids, 100, density=True, align='mid')
# X = np.linspace(min(bins), max(bins), 200)
# # pdf = (np.exp(-(np.log(X) - 3)**2 / (2 * 1**2))
# #        / (X * 1 * np.sqrt(2 * np.pi)))
# cdf = 1/2 * ( 1 + sp.erf((np.log(X) - 3) / (np.sqrt(2) * 1)) )
# # plt.plot(X, cdf, linewidth=2, color='r')
# plt.axis('tight')
# plt.show()


F_U_array = []
F_L_array = []
X2 = np.linspace(0, 200, 100)
for v in tqdm(X2):
    F_U_v = HTE.F_hat_U(max_bid_dicts_top2only,HTE.calc_M(max_bid_dicts_top2only),v,0.1)
    F_U_array.append(F_U_v)
    F_L_v = HTE.F_hat_L(max_bid_dicts_top2only,HTE.calc_M(max_bid_dicts_top2only),v,-0.1,increment=1)
    F_L_array.append(F_L_v)

print(F_U_array)
print(F_L_array)

plt.scatter(X2,F_U_array,color='orange')
plt.scatter(X2,F_L_array,color='blue')
cdf = 1/2 * ( 1 + sp.erf((np.log(X2) - 3) / (np.sqrt(2) * 1)) )
plt.plot(X2, cdf, linewidth=2, color='r')
plt.show()

