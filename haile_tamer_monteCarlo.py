import haile_tamer_simulation as HTS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def run_Tn_simulations(Tn, n, lbda, increment, distribution='lognormal'):
    all_values = []
    all_bids = []
    for i in range(Tn):
        values, bids, irrelevant = HTS.run_1_simulation(n, lbda, increment, distribution)
        all_values.extend(values)
        all_bids.extend(bids)

    return all_values, all_bids

all_values, all_bids = run_Tn_simulations(Tn=100, n=6, lbda=0.1, increment=0.1, distribution='lognormal')

# Plot Value & Bid Distributions
count, bins, ignored = plt.hist(all_bids, 100, density=True, align='mid')
x = np.linspace(min(bins), max(bins), 10000)
pdf = (np.exp(-(np.log(x) - 3)**2 / (2 * 1**2))
       / (x * 1 * np.sqrt(2 * np.pi)))
plt.plot(x, pdf, linewidth=2, color='r')
plt.axis('tight')
plt.show()