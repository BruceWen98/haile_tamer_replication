import pandas as pd
import numpy as np
import non_equilibrium_profits as NEP
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import sys
import os
from pathlib import Path

def compute_v0(dicts):
    ratios = []
    for d in dicts:
        ratios.append(d['low_rel_high'])
    return np.median(ratios)

def compute_exp_profit(X,n,data_dicts,v0):
    profits = []
    v_nn, ghat_KDE_nn, _ = NEP.ghat_KDE(data_dicts, n, n, ker='gaussian', bandwidth='ISJ')
    G_hat_vnn = NEP.KDE_pdf_to_cdf(v_nn, ghat_KDE_nn)
    for r in tqdm(X):
        p = NEP.compute_exp_profit_non_equil(n, r, data_dicts,v_nn,ghat_KDE_nn,G_hat_vnn, v0)
        profits.append(p)
    return profits

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
            
    print("Values of N = ", NEP.calc_N_set(data_dicts))
    print("Size of this Data = ", len(data_dicts))

    v0 = compute_v0(data_dicts)
    X = np.linspace(0, UB_V, num_points)

    ### 3. Profit against reserve price curve for each n 
    allbounds    = []
    print("n = {}".format(n))
    print(len([d for d in data_dicts if d['n'] == n]))

    try:
        profits = compute_exp_profit(X,n, data_dicts, v0)
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
    plt.plot(X,profits,color='tab:blue',linewidth=2, label='lb')    # marker=6 is a caretup
    plt.savefig(OUTPATH + "profits_n{}_nonEquilibrium.png".format(n))
    return


f = sys.argv[1]
print("Working on {} now...".format(f))
INPATH = str(f)
OUTPATH = "/Users/brucewen/Desktop/honors_thesis/estimation/combined_data/categorized_data_results_withXi/" + INPATH.split("/")[-1].split(".")[0] + "/"
SELECT_RESERVE_PATH = "/Users/brucewen/Desktop/honors_thesis/selecting reserve/code/computed_bounds/"
if Path(OUTPATH).is_dir()==False:
    os.mkdir(OUTPATH)


run_AL_profits_specificN_Data(INPATH, OUTPATH, n=3, UB_V=10, num_points=500)