import pandas as pd
import numpy as np
import AL_profits as AL
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 15})

import pickle
import seaborn as sns
import AL_unconditionalN as ALUN
import sys
import os
from pathlib import Path

### 1. Helping Functions
def compute_v0(dicts):
    ratios = []
    for d in dicts:
        ratios.append(d['low_rel_high'])
    return np.median(ratios)

def compute_bounds(X,n, data_dicts, UB_V, V0):
    profits_lb_AL = []
    profits_ub_AL = []
    ci_lbs = []
    ci_ubs = []
    for r in tqdm(X):
        p_AL_lb, p_AL_ub, ci_lb, ci_ub = AL.compute_expected_profit_KDE(n, r, data_dicts, UB_V, V0, variedN=False)
        profits_lb_AL.append(p_AL_lb)
        profits_ub_AL.append(p_AL_ub)
        ci_lbs.append(ci_lb)
        ci_ubs.append(ci_ub)
    return profits_lb_AL, profits_ub_AL, ci_lbs, ci_ubs
        
def compute_bounds_vN(X,n, data_dicts, UB_V, V0, KDEs):
    profits_lb_AL_vN = []
    profits_ub_AL_vN = []
    ci_lbs = []
    ci_ubs = []
    for r in tqdm(X):
        # p_AL_lb, p_AL_ub = AL.compute_expected_profit_KDE(n, r, data_dicts, UB_V, V0, variedN=True, KDEs=KDEs)
        p_AL_lb, p_AL_ub, ci_lb, ci_ub = AL.compute_expected_profit_KDE(n, r, data_dicts, UB_V, V0, variedN=True, KDEs=KDEs)
        profits_lb_AL_vN.append(p_AL_lb)
        profits_ub_AL_vN.append(p_AL_ub)
        ci_lbs.append(ci_lb)
        ci_ubs.append(ci_ub)
    return profits_lb_AL_vN, profits_ub_AL_vN, ci_lbs, ci_ubs

def compute_bounds_vN_barN(X,n, barN, data_dicts, UB_V, V0, KDEs):
    profits_lb_AL_vN = []
    profits_ub_AL_vN = []
    for r in tqdm(X):
        # p_AL_lb, p_AL_ub = AL.compute_expected_profit_KDE(n, r, data_dicts, UB_V, V0, variedN=True, KDEs=KDEs)
        p_AL_lb, p_AL_ub = AL.compute_expected_profit_KDE_barN(n, r, data_dicts, UB_V, V0, barN, variedN=True, KDEs=KDEs)
        profits_lb_AL_vN.append(p_AL_lb)
        profits_ub_AL_vN.append(p_AL_ub)
    return profits_lb_AL_vN, profits_ub_AL_vN

def getNamesForGraphs2(characteristics):
    # Category
    if characteristics['cat0']==None and characteristics['cat1']==None:
        cat = 'all'
    elif characteristics['cat0']!=None and characteristics['cat1']==None:
        cat = characteristics['cat0']
    else:
        cat = characteristics['cat0'] + ':' + characteristics['cat1']
    
    # Location
    if characteristics['loc'] == None:
        loc = 'all'
    else:
        loc = characteristics['loc'] 
    
    # Value
    if characteristics['value'] == None:
        val = 'all'
    else:
        val = characteristics['value']
    
    # Degree of Competition
    if characteristics['degree_competition'] == None:
        deg_competition = 'all'
    else:
        deg_competition = characteristics['degree_competition']
        
    return cat, loc, val, deg_competition

def run_AL_profits_specificN_Data(INPATH, OUTPATH, n, UB_V=15, num_points=1000):
    ### 2. Input Data
    with open(INPATH, 'rb') as handle:
        data_file = pickle.load(handle)
        data_dicts = data_file['data']
        characteristics = data_file['characteristics']
        
    cat, loc, val, deg_competition = getNamesForGraphs2(characteristics)
            
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
            profits_lb_AL, profits_ub_AL, ci_lbs, ci_ubs = compute_bounds(X,n, data_dicts, UB_V, V0)
        except ValueError as e:     #ValueError: Root finding did not converge. Need more data.
            print(e)
            return
    else:
        try: 
            profits_lb_AL, profits_ub_AL, ci_lbs, ci_ubs = compute_bounds(X,n, data_dicts, UB_V, V0)
            profits_lb_AL_vN, profits_ub_AL_vN, ci_lbs_vN, ci_ubs_vN = compute_bounds_vN(X,n, data_dicts, UB_V, V0, KDEs)
        except ValueError as e:     #ValueError: Root finding did not converge. Need more data.
            print(e)
            return 

        obs = len([d for d in data_dicts if d['n'] == n])
        sns.set_style('darkgrid')
        plt.figure(figsize=(10,6), tight_layout=True)
        plt.title("Category: {}, Loc: {}, Sell Price: {}, Competition: {};  N={}; Obs={}".format(cat,loc,val,deg_competition,n,obs), fontsize=13)
        plt.xlabel("Reserve Price")
        plt.ylabel("Expected Profit")
        plt.ylim(-0.1, 1)
        plt.plot(X,profits_lb_AL,color='tab:blue',linewidth=2, label='lb')    # marker=6 is a caretup
        plt.plot(X,profits_ub_AL,color='tab:blue',linewidth=2, label='ub')      # marker=7 is a caretdown
        plt.plot(X,ci_lbs,color='lightsteelblue', linestyle='dashed', label='95% CI lb')
        plt.plot(X,ci_ubs,color='lightsteelblue', linestyle='dashed', label='95% CI ub')
        plt.savefig(OUTPATH + "profits_n{}.png".format(n))
        if profits_lb_AL_vN is not None:    # plot the 2 plots together.
            plt.plot(X,profits_lb_AL_vN,color='#9467bd',linewidth=2, label='lb varying N')    # marker=6 is a caretup
            plt.plot(X,profits_ub_AL_vN,color='#9467bd',linewidth=2, label='ub varying N')      # marker=7 is a caretdown
            plt.plot(X,ci_lbs_vN,color='#c5b0d5', linestyle='dashed', label='95% CI lb varying N')
            plt.plot(X,ci_ubs_vN,color='#c5b0d5', linestyle='dashed', label='95% CI ub varying N')
            plt.savefig(OUTPATH + "profits_n{}_combined.png".format(n))
        
        plt.close()
        if profits_lb_AL_vN is not None:
            sns.set_style('darkgrid')
            plt.figure(figsize=(10,6), tight_layout=True)
            plt.title("Category: {}, Loc: {}, Sell Price: {}, Competition: {};  N={}; Obs={}".format(cat,loc,val,deg_competition,n,obs), fontsize=13)
            plt.xlabel("Reserve Price")
            plt.ylabel("Expected Profit")
            plt.ylim(-0.1, 1)
            plt.plot(X,profits_lb_AL_vN,color='#9467bd',linewidth=2, label='lb varying N')    # marker=6 is a caretup
            plt.plot(X,profits_ub_AL_vN,color='#9467bd',linewidth=2, label='ub varying N')      # marker=7 is a caretdown
            plt.plot(X,ci_lbs_vN,color='#c5b0d5', linestyle='dashed', label='95% CI lb varying N')
            plt.plot(X,ci_ubs_vN,color='#c5b0d5', linestyle='dashed', label='95% CI ub varying N')
            allbounds_vN.append({'N': n, 
                        'lb': profits_lb_AL_vN, 
                        'ub': profits_ub_AL_vN, 
                        'lb_95': ci_lbs_vN,
                        'ub_95': ci_ubs_vN,
                        'count': len([d for d in data_dicts if d['n'] == n])})
            plt.savefig(OUTPATH + "profits_n{}_vN.png".format(n))
            plt.close()

    allbounds.append({'N': n, 'lb': profits_lb_AL, 'ub': profits_ub_AL, 'count': len([d for d in data_dicts if d['n'] == n])})

    return X, allbounds_vN

def run_AL_profits_specificData(INPATH, OUTPATH, RESERVE_PATH, UB_V=15, num_points=1000):
    with open(INPATH, 'rb') as handle:
        data_file = pickle.load(handle)
        data_dicts = data_file['data']
        characteristics = data_file['characteristics']
        
    cat, loc, val, deg_competition = getNamesForGraphs2(characteristics)
            
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
                profits_lb_AL, profits_ub_AL, ci_lbs, ci_ubs = compute_bounds(X,n, data_dicts, UB_V, V0)
            except ValueError as e:     #ValueError: Root finding did not converge. Need more data.
                print(e)
                return
        else:
            try: 
                profits_lb_AL, profits_ub_AL, ci_lbs, ci_ubs = compute_bounds(X,n, data_dicts, UB_V, V0)
                profits_lb_AL_vN, profits_ub_AL_vN, ci_lbs_vN, ci_ubs_vN = compute_bounds_vN(X,n, data_dicts, UB_V, V0, KDEs)
            except ValueError as e:     #ValueError: Root finding did not converge. Need more data.
                print(e)
                return 


        obs = len([d for d in data_dicts if d['n'] == n])
        sns.set_style('darkgrid')
        plt.figure(figsize=(10,6), tight_layout=True)
        plt.title("Category: {}, Loc: {}, Sell Price: {}, Competition: {};  N={}; Obs={}".format(cat,loc,val,deg_competition,n,obs), fontsize=13)
        plt.xlabel("Reserve Price")
        plt.ylabel("Expected Profit")
        # plt.ylim(-0.05, 1.5)
        plt.plot(X,profits_lb_AL,color='tab:blue',linewidth=2, label='lb')    # marker=6 is a caretup
        plt.plot(X,profits_ub_AL,color='tab:blue',linewidth=2, label='ub')      # marker=7 is a caretdown
        plt.plot(X,ci_lbs,color='lightsteelblue', linestyle='dashed', label='95% CI lb')
        plt.plot(X,ci_ubs,color='lightsteelblue', linestyle='dashed', label='95% CI ub')
        plt.savefig(OUTPATH + "profits_n{}.png".format(n))
        
        if profits_lb_AL_vN is not None:    # plot the 2 plots together.
            plt.plot(X,profits_lb_AL_vN,color='#9467bd',linewidth=2, label='lb varying N')    # marker=6 is a caretup
            plt.plot(X,profits_ub_AL_vN,color='#9467bd',linewidth=2, label='ub varying N')      # marker=7 is a caretdown
            plt.plot(X,ci_lbs_vN,color='#c5b0d5', linestyle='dashed', label='95% CI lb varying N')
            plt.plot(X,ci_ubs_vN,color='#c5b0d5', linestyle='dashed', label='95% CI ub varying N')
            plt.savefig(OUTPATH + "profits_n{}_combined.png".format(n))
        
        plt.close()
        if profits_lb_AL_vN is not None:
            sns.set_style('darkgrid')
            plt.figure(figsize=(10,6), tight_layout=True)
            plt.title("Category: {}, Loc: {}, Sell Price: {}, Competition: {};  N={}; Obs={}".format(cat,loc,val,deg_competition,n,obs), fontsize=13)
            plt.xlabel("Reserve Price")
            plt.ylabel("Expected Profit")
            # plt.ylim(-0.05, 1.5)
            plt.plot(X,profits_lb_AL_vN,color='#9467bd',linewidth=2, label='lb varying N')    # marker=6 is a caretup
            plt.plot(X,profits_ub_AL_vN,color='#9467bd',linewidth=2, label='ub varying N')      # marker=7 is a caretdown
            plt.plot(X,ci_lbs_vN,color='#c5b0d5', linestyle='dashed', label='95% CI lb varying N')
            plt.plot(X,ci_ubs_vN,color='#c5b0d5', linestyle='dashed', label='95% CI ub varying N')
            allbounds_vN.append({'N': n, 
                                 'lb': profits_lb_AL_vN, 
                                 'ub': profits_ub_AL_vN, 
                                 'lb_95': ci_lbs_vN,
                                 'ub_95': ci_ubs_vN,
                                 'count': len([d for d in data_dicts if d['n'] == n])})
            plt.savefig(OUTPATH + "profits_n{}_vN.png".format(n))
            plt.close()
            
        # plt.legend()
        # plt.show()
        # plt.savefig(OUTPATH + "profits_n{}.png".format(n))
        
        allbounds.append({'N': n, 'lb': profits_lb_AL, 'ub': profits_ub_AL, 'count': len([d for d in data_dicts if d['n'] == n])})

    return X, allbounds_vN


f = sys.argv[1]

print("Working on {} now...".format(f))
INPATH = str(f)
OUTPATH = "/Users/brucewen/Desktop/honors_thesis/estimation/combined_data/categorized_data_results_withXi/" + INPATH.split("/")[-1].split(".")[0] + "/"
SELECT_RESERVE_PATH = "/Users/brucewen/Desktop/honors_thesis/selecting reserve/code/computed_bounds/"
if Path(OUTPATH).is_dir()==False:
    os.mkdir(OUTPATH)


print(OUTPATH)

# X, allbounds_vN = run_AL_profits_specificData(INPATH, OUTPATH, SELECT_RESERVE_PATH, UB_V=10, num_points=500)
# out = (X, allbounds_vN)
# pickle.dump(out, open(OUTPATH + "allbounds_vN.p", "wb"))

X, bounds_vN_n3 = run_AL_profits_specificN_Data(INPATH, OUTPATH, n=3, UB_V=10, num_points=500)
out = (X, bounds_vN_n3)
pickle.dump(out, open(OUTPATH + "bounds_vN_n3.p", "wb"))
