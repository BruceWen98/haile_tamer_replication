import haile_tamer_simulation as HTS
import haile_tamer_estimation_withN as HTE
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
from tqdm import tqdm
import collections

def run_T_simulations_varyingN(T, lbda, increment, distribution='lognormal'):
    all_values = []
    all_bids = []
    max_bid_dicts = []
    for i in range(T):
        n = np.random.randint(2, 10)
        values, bids, losing_player, max_bid_dict = HTS.run_1_simulation(n, lbda, increment, distribution)
        all_values.append(values)
        all_bids.append(bids)
        
        max_bid_dict = dict(collections.OrderedDict(sorted(max_bid_dict.items())))  # Sort the dict by key, ascending order.
        max_bid_dicts.append(max_bid_dict)

    return all_values, all_bids, max_bid_dicts

all_values, all_bids, max_bid_dicts = run_T_simulations_varyingN(T=400, lbda=0.1, increment=0.1, distribution='lognormal')


# What if we only took the top 2 bids? Even though there are >2 bidders in each auction.
def top2(d):
    top2keys = sorted(d, key=d.get, reverse=True)[:2]
    out_dict = {}
    out_dict["1"] = d[top2keys[0]]
    out_dict["2"] = d[top2keys[1]]
    out_dict["n"] = len(d)
    return out_dict

max_bid_dicts_top2only = [top2(max_bid_dict) for max_bid_dict in max_bid_dicts]
print(max_bid_dicts_top2only)


F_U_array = []
F_L_array = []
X2 = np.linspace(0, 200, 100)
for v in tqdm(X2):
    F_U_v = HTE.F_hat_U(max_bid_dicts_top2only,v,0.1)
    F_U_array.append(F_U_v)
    F_L_v = HTE.F_hat_L(max_bid_dicts_top2only,v,-0.1,increment=0.1)
    F_L_array.append(F_L_v)

print(F_U_array)
print(F_L_array)

plt.scatter(X2,F_U_array,color='orange')
plt.scatter(X2,F_L_array,color='blue')
cdf = 1/2 * ( 1 + sp.erf((np.log(X2) - 3) / (np.sqrt(2) * 1)) )
plt.plot(X2, cdf, linewidth=2, color='r')
plt.show()

