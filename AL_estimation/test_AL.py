import pandas as pd
import numpy as np
import AL_profits as AL
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle

VAL = "highVal"

with open('/Users/brucewen/Desktop/honors_thesis/estimation/sotheby_data/with_N_bidders/output_dicts_ALL_{}.pickle'.format(VAL), 'rb') as handle:
    output_dicts_ALL = pickle.load(handle)


print(output_dicts_ALL)
X = np.linspace(0, 15, 100)


### Profit against reserve price curve
n=2
profits_lb = []
profits_ub = []
for r in tqdm(X):
    h = X[1] - X[0]
    p_lb, p_ub = AL.compute_expected_profit(n, r, output_dicts_ALL, 15, 1, h, variedN=True)
    profits_lb.append(p_lb)
    profits_ub.append(p_ub)
    
plt.scatter(X,profits_lb,color='orange', label='profits low bound', marker='_')
plt.scatter(X,profits_ub,color='blue', label='profits high bound', marker='|')
plt.show()