import AL_profits as AL
import profit_diff as PD
import draw_AL_profits_specific as DAPS
import draw_non_equilibrium_profits as DNEP
import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from pathlib import Path

def compute_profit_diff_bounds(X,n,data_dicts,v0):
    diff_lb = []
    diff_ub = []
    b_nns, b_n1ns = DAPS.get_bnns_bn1ns(data_dicts, n)
    
    _,_,v_nn,_,_,G_hat_vnn = DAPS.get_KDE_parameters(data_dicts, n)
    
    for r in tqdm(X):
        lb, ub = PD.compute_expected_profit_diff(n,r,v0,
                                                 v_nn, G_hat_vnn,
                                                 b_nns, b_n1ns)
        diff_lb.append(lb)
        diff_ub.append(ub)
    
    return diff_lb, diff_ub

def compute_profit_diff_bounds_usingAL(X,n,data_dicts,v0):
    diff_lb = []
    diff_ub = []
    b_nns, b_n1ns = DAPS.get_bnns_bn1ns(data_dicts, n)
    
    _,_,v_nn,_,_,G_hat_vnn = DAPS.get_KDE_parameters(data_dicts, n)
    for r in tqdm(X):
        lb, ub = PD.compute_expected_profit_diffAL(n,r,v0,
                                                    v_nn,G_hat_vnn,
                                                    b_nns, b_n1ns,
                                                    )
        diff_lb.append(lb)
        diff_ub.append(ub)
    
    return diff_lb, diff_ub

def compute_exact_profit_diff(X,n,data_dicts,v0):
    diffs = []
    
    _,_,v_nn,ghat_KDE_nn,_,G_hat_vnn = DAPS.get_KDE_parameters(data_dicts, n)
    
    for r in X:
        d = PD.exact_profit_diff(n,r,data_dicts,
                                 v_nn, ghat_KDE_nn, G_hat_vnn,
                                 v0)
        diffs.append(d)
    
    return diffs

def draw_profit_diff_specificN(INPATH, OUTPATH, n, ub_v=10, num_points=1000):
    ### 2. Input Data
    with open(INPATH, 'rb') as handle:
        data_file = pickle.load(handle)
        data_dicts = data_file['data']
        characteristics = data_file['characteristics']
        
    cat, loc, val, deg_competition = DAPS.getNamesForGraphs2(characteristics)
            
    print("Values of N = ", AL.calc_N_set(data_dicts))
    print("Size of this Data = ", len(data_dicts))

    v0 = DAPS.compute_v0(data_dicts)
    X = np.linspace(0, ub_v, num_points)
    
    print("n = {}".format(n))
    print(len([d for d in data_dicts if d['n'] == n]))

    try:
        diff_lb, diff_ub = compute_profit_diff_bounds(X,n,data_dicts,v0)
        diff = compute_exact_profit_diff(X,n, data_dicts, v0)
    except ValueError as e:     #ValueError: Root finding did not converge. Need more data.
        print(e)
        return
    
    obs = len([d for d in data_dicts if d['n'] == n])
    sns.set_style('darkgrid')
    plt.figure(figsize=(10,6), tight_layout=True)
    plt.title("Category: {}, Loc: {}, Sell Price: {}, Competition: {};  N={}; Obs={}".format(cat,loc,val,deg_competition,n,obs), fontsize=13)
    plt.xlabel("Reserve Price")
    plt.ylabel(r'Expected Profit Difference relative to $v_0$')
    # plt.ylim(-0.1, 1)
    plt.plot(X,diff,color='black',linewidth=2,linestyle='dashdot', label='non-equilibrium')
    plt.plot(X,diff_ub,color='tab:blue',linewidth=2, label='ub')
    plt.plot(X,diff_lb,color='tab:blue',linewidth=2, label='lb')
    plt.legend()
    plt.savefig(OUTPATH + "profit_diff_n{}.png".format(n))
    return


