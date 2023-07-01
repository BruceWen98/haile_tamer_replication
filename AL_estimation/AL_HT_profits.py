import pandas as pd
import numpy as np
import AL_profits as AL
import HT_profits as HT
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import seaborn as sns

VAL = "withN"
OUTPATH = "/Users/brucewen/Desktop/honors_thesis/estimation/AL_estimation/figs/"

with open('/Users/brucewen/Desktop/honors_thesis/estimation/sotheby_data/with_N_bidders/output_dicts_ALL_{}.pickle'.format(VAL), 'rb') as handle:
    output_dicts_ALL = pickle.load(handle)


print(output_dicts_ALL)
X = np.linspace(0, 15, 100)


### Profit against reserve price curve
# n=2

for n in list(np.unique([d['n'] for d in output_dicts_ALL]))[7:]:
    print("n = {}".format(n))
    profits_lb_AL = []
    profits_ub_AL = []
    profits_lb_HT = []
    profits_ub_HT = []
    for r in tqdm(X):
        p_AL_lb, p_AL_ub = AL.compute_expected_profit_LLN(n, r, output_dicts_ALL, 15, 1.1, variedN=False)
        p_HT_lb, p_HT_ub = HT.compute_expected_profit_LLN(n, r, output_dicts_ALL, 15, 1.1)
        profits_lb_AL.append(p_AL_lb)
        profits_ub_AL.append(p_AL_ub)
        profits_lb_HT.append(p_HT_lb)
        profits_ub_HT.append(p_HT_ub)

    sns.set_style('darkgrid')
    plt.figure(figsize=(10,6), tight_layout=True)
    plt.title("Expected Profit against reserve price curve")
    plt.xlabel("Reserve Price")
    plt.ylabel("Expected Profit")
    plt.scatter(X,profits_lb_AL,color='orange', label='AL lb', marker='_')
    plt.scatter(X,profits_ub_AL,color='blue', label='AL hb', marker='|')
    plt.scatter(X,profits_lb_HT,color='purple', label='HT lb', marker='x')
    plt.scatter(X,profits_ub_HT,color='pink', label='HT ub', marker='^')

    plt.legend()
    # plt.show()
    plt.savefig(OUTPATH + "profits_n{}.png".format(n))