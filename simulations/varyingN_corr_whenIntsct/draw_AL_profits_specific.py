import pandas as pd
import numpy as np
import AL_profits as AL
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import sys
import os
from pathlib import Path

### 1. Helping Functions
def compute_v0(dicts):
    return 5.0

def compute_bounds(X,n, data_dicts, UB_V, V0):
    profits_lb_AL = []
    profits_ub_AL = []
    for r in tqdm(X):
        p_AL_lb, p_AL_ub = AL.compute_expected_profit_KDE(n, r, data_dicts, UB_V, V0, variedN=False)
        profits_lb_AL.append(p_AL_lb)
        profits_ub_AL.append(p_AL_ub)
    return profits_lb_AL, profits_ub_AL
        
def compute_bounds_vN(X,n, data_dicts, UB_V, V0, KDEs):
    profits_lb_AL_vN = []
    profits_ub_AL_vN = []
    for r in tqdm(X):
        # p_AL_lb, p_AL_ub = AL.compute_expected_profit_KDE(n, r, data_dicts, UB_V, V0, variedN=True, KDEs=KDEs)
        p_AL_lb, p_AL_ub = AL.compute_expected_profit_KDE(n, r, data_dicts, UB_V, V0, variedN=True, KDEs=KDEs)
        profits_lb_AL_vN.append(p_AL_lb)
        profits_ub_AL_vN.append(p_AL_ub)
    return profits_lb_AL_vN, profits_ub_AL_vN

def compute_bounds_vN_barN(X,n, barN, data_dicts, UB_V, V0, KDEs):
    profits_lb_AL_vN = []
    profits_ub_AL_vN = []
    for r in tqdm(X):
        # p_AL_lb, p_AL_ub = AL.compute_expected_profit_KDE(n, r, data_dicts, UB_V, V0, variedN=True, KDEs=KDEs)
        p_AL_lb, p_AL_ub = AL.compute_expected_profit_KDE_barN(n, r, data_dicts, UB_V, V0, barN, variedN=True, KDEs=KDEs)
        profits_lb_AL_vN.append(p_AL_lb)
        profits_ub_AL_vN.append(p_AL_ub)
    return profits_lb_AL_vN, profits_ub_AL_vN


def run_AL_profits_specificN_Data(INPATH, OUTPATH, n, UB_V=15, num_points=1000):
    ### 2. Input Data
    with open(INPATH, 'rb') as handle:
        data_dicts = pickle.load(handle)

    print("Values of N = ", AL.calc_N_set(data_dicts))
    print("Size of this Data = ", len(data_dicts))

    V0 = compute_v0(data_dicts)
    X = np.linspace(V0, UB_V, num_points)

    KDEs = AL.getAll_KDEs(2,data_dicts)     # comment if not using variedN


    ### 3. Profit against reserve price curve for each n 
    allbounds    = []
    allbounds_vN = []
    print("n = {}".format(n))
    print(len([d for d in data_dicts if d['n'] == n]))
    profits_lb_AL_vN, profits_ub_AL_vN = None, None
    if n == max([d['n'] for d in data_dicts]):
        try:
            profits_lb_AL, profits_ub_AL = compute_bounds(X,n, data_dicts, UB_V, V0)
        except ValueError as e:     #ValueError: Root finding did not converge. Need more data.
            print(e)
            return
    else:
        try: 
            profits_lb_AL, profits_ub_AL = compute_bounds(X,n, data_dicts, UB_V, V0)
            profits_lb_AL_vN, profits_ub_AL_vN = compute_bounds_vN(X,n, data_dicts, UB_V, V0, KDEs)
        except ValueError as e:     #ValueError: Root finding did not converge. Need more data.
            print(e)
            return 

    sns.set_style('darkgrid')
    plt.figure(figsize=(10,6), tight_layout=True)
    plt.title("Expected Profit against Reserve Price, N = {}, Intersecting N".format(n))
    plt.xlabel("Reserve Price")
    plt.ylabel("Expected Profit")
    plt.scatter(X,profits_lb_AL,color='plum', label='lb', marker=6)    # marker=6 is a caretup
    plt.scatter(X,profits_ub_AL,color='darkorchid', label='ub', marker=7)      # marker=7 is a caretdown
    if profits_lb_AL_vN is not None:
        plt.scatter(X,profits_lb_AL_vN,color='limegreen', label='lb varying N', marker=6)    # marker=6 is a caretup
        plt.scatter(X,profits_ub_AL_vN,color='seagreen', label='ub varying N', marker=7)      # marker=7 is a caretdown
        allbounds_vN.append({'N': n, 'lb': profits_lb_AL_vN, 'ub': profits_ub_AL_vN, 'count': len([d for d in data_dicts if d['n'] == n])})

    plt.legend()
    # plt.show()
    plt.savefig(OUTPATH + "profits_n{}_IntersectingN.png".format(n))

    allbounds.append({'N': n, 'lb': profits_lb_AL, 'ub': profits_ub_AL, 'count': len([d for d in data_dicts if d['n'] == n])})

    return

def run_AL_profits_specificN_Data_barN(INPATH, OUTPATH, n, barN, UB_V=15, num_points=1000):
    ### 2. Input Data
    with open(INPATH, 'rb') as handle:
        data_dicts = pickle.load(handle)

    print("Values of N = ", AL.calc_N_set(data_dicts))
    print("Size of this Data = ", len(data_dicts))

    V0 = compute_v0(data_dicts)
    X = np.linspace(V0, UB_V, num_points)

    KDEs = AL.getAll_KDEs(2,data_dicts)     # comment if not using variedN


    ### 3. Profit against reserve price curve for each n 
    allbounds    = []
    allbounds_vN = []
    print("n = {}".format(n))
    print(len([d for d in data_dicts if d['n'] == n]))
    profits_lb_AL_vN, profits_ub_AL_vN = None, None
    if n == max([d['n'] for d in data_dicts]):
        try:
            profits_lb_AL, profits_ub_AL = compute_bounds(X,n, data_dicts, UB_V, V0)
        except ValueError as e:     #ValueError: Root finding did not converge. Need more data.
            print(e)
            return
    else:
        try: 
            profits_lb_AL, profits_ub_AL = compute_bounds(X,n, data_dicts, UB_V, V0)
            profits_lb_AL_vN, profits_ub_AL_vN = compute_bounds_vN_barN(X,n,barN, data_dicts, UB_V, V0, KDEs)
        except ValueError as e:     #ValueError: Root finding did not converge. Need more data.
            print(e)
            return 

    sns.set_style('darkgrid')
    plt.figure(figsize=(10,6), tight_layout=True)
    plt.title("Expected Profit against Reserve Price, N = {}, barN= {}".format(n, barN))
    plt.xlabel("Reserve Price")
    plt.ylabel("Expected Profit")
    plt.scatter(X,profits_lb_AL,color='plum', label='lb', marker=6)    # marker=6 is a caretup
    plt.scatter(X,profits_ub_AL,color='darkorchid', label='ub', marker=7)      # marker=7 is a caretdown
    if profits_lb_AL_vN is not None:
        plt.scatter(X,profits_lb_AL_vN,color='limegreen', label='lb varying N', marker=6)    # marker=6 is a caretup
        plt.scatter(X,profits_ub_AL_vN,color='seagreen', label='ub varying N', marker=7)      # marker=7 is a caretdown
        allbounds_vN.append({'N': n, 'lb': profits_lb_AL_vN, 'ub': profits_ub_AL_vN, 'count': len([d for d in data_dicts if d['n'] == n])})

    plt.legend()
    # plt.show()
    plt.savefig(OUTPATH + "profits_n{}_barN{}.png".format(n, barN))

    allbounds.append({'N': n, 'lb': profits_lb_AL, 'ub': profits_ub_AL, 'count': len([d for d in data_dicts if d['n'] == n])})

    return

def run_AL_profits_specificData(INPATH, OUTPATH, RESERVE_PATH, UB_V=15, num_points=1000):
    ### 2. Input Data
    with open(INPATH, 'rb') as handle:
        data_dicts = pickle.load(handle)

    print("Values of N = ", AL.calc_N_set(data_dicts))
    print("Size of this Data = ", len(data_dicts))

    V0 = compute_v0(data_dicts)
    X = np.linspace(V0, UB_V, num_points)

    KDEs = AL.getAll_KDEs(2,data_dicts)     # comment if not using variedN


    ### 3. Profit against reserve price curve for each n 
    allbounds    = []
    allbounds_vN = []
    for n in list(np.unique([d['n'] for d in data_dicts])):
        print("n = {}".format(n))
        print(len([d for d in data_dicts if d['n'] == n]))
        profits_lb_AL_vN, profits_ub_AL_vN = None, None
        if n == max([d['n'] for d in data_dicts]):
            try:
                profits_lb_AL, profits_ub_AL = compute_bounds(X,n, data_dicts, UB_V, V0)
            except ValueError as e:     #ValueError: Root finding did not converge. Need more data.
                print(e)
                continue
        else:
            try: 
                profits_lb_AL, profits_ub_AL = compute_bounds(X,n, data_dicts, UB_V, V0)
                profits_lb_AL_vN, profits_ub_AL_vN = compute_bounds_vN(X,n, data_dicts, UB_V, V0, KDEs)
            except ValueError as e:     #ValueError: Root finding did not converge. Need more data.
                print(e)
                continue

        sns.set_style('darkgrid')
        plt.figure(figsize=(10,6), tight_layout=True)
        plt.title("Expected Profit against Reserve Price, N = {}".format(n))
        plt.xlabel("Reserve Price")
        plt.ylabel("Expected Profit")
        plt.scatter(X,profits_lb_AL,color='plum', label='lb', marker=6)    # marker=6 is a caretup
        plt.scatter(X,profits_ub_AL,color='darkorchid', label='ub', marker=7)      # marker=7 is a caretdown
        if profits_lb_AL_vN is not None:
            plt.scatter(X,profits_lb_AL_vN,color='limegreen', label='lb varying N', marker=6)    # marker=6 is a caretup
            plt.scatter(X,profits_ub_AL_vN,color='seagreen', label='ub varying N', marker=7)      # marker=7 is a caretdown
            allbounds_vN.append({'N': n, 'lb': profits_lb_AL_vN, 'ub': profits_ub_AL_vN, 'count': len([d for d in data_dicts if d['n'] == n])})

        plt.legend()
        # plt.show()
        plt.savefig(OUTPATH + "profits_n{}.png".format(n))

        allbounds.append({'N': n, 'lb': profits_lb_AL, 'ub': profits_ub_AL, 'count': len([d for d in data_dicts if d['n'] == n])})

    return


f = sys.argv[1]

print("Working on {} now...".format(f))
INPATH = str(f)
OUTPATH = "/Users/brucewen/Desktop/honors_thesis/estimation/simulations/varyingN_corr_whenIntsct/results/" + INPATH.split("/")[-1].rsplit(".",1)[0] + "/"
if Path(OUTPATH).is_dir()==False:
    os.mkdir(OUTPATH)


print(OUTPATH)
# run_AL_profits_specificData(INPATH, OUTPATH, SELECT_RESERVE_PATH, UB_V=5, num_points=200)

# run_AL_profits_specificN_Data(INPATH, OUTPATH, n=3, UB_V=15, num_points=200)

for x in [4,5,6,7,8,9,10,11]:
    run_AL_profits_specificN_Data_barN(INPATH, OUTPATH, n=3, barN=x, UB_V=15, num_points=200)