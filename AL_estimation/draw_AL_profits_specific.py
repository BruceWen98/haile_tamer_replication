import pandas as pd
import numpy as np
import AL_profits as AL
import AL_nonequilibrium as AL_NEP
import non_equilibrium_profits as NEP
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 15})

import pickle
import seaborn as sns


### 1. Helping Functions
def compute_v0(dicts):
    ratios = []
    for d in dicts:
        ratios.append(d['low_rel_high'])
    return np.median(ratios)

def get_KDE_parameters(data_dicts, n):
    v_n1n, ghat_KDE_n1n, h = AL.ghat_KDE(data_dicts,n-1,n, ker='gaussian', bandwidth="ISJ")
    v_nn, ghat_KDE_nn, h = AL.ghat_KDE(data_dicts,n,n, ker='gaussian', bandwidth="ISJ")
    G_hat_vn1n = AL.KDE_pdf_to_cdf(v_n1n, ghat_KDE_n1n)
    G_hat_vnn = AL.KDE_pdf_to_cdf(v_nn, ghat_KDE_nn)
    return v_n1n, ghat_KDE_n1n, v_nn, ghat_KDE_nn, G_hat_vn1n, G_hat_vnn

def get_bnns_bn1ns(data_dicts, n):
    # Get the b_nns and b_n1ns
    b_nns = []
    b_n1ns = []
    for data_dict in data_dicts:
        if data_dict['n'] == n:
            b_nn = max( data_dict.get('1'), data_dict.get('2') )
            b_n1n = min( data_dict.get('1'), data_dict.get('2') )
            b_nns.append(b_nn)
            b_n1ns.append(b_n1n)
    return b_nns, b_n1ns

def compute_bounds(X,n, data_dicts, ub_v, v0):
    profits_lb_AL = []
    profits_ub_AL = []
    ci_lbs = []
    ci_ubs = []
    
    b_nns, b_n1ns = get_bnns_bn1ns(data_dicts, n)
    
    # Get the KDE parameters
    v_n1n, ghat_KDE_n1n, v_nn, ghat_KDE_nn, _, G_hat_vnn = get_KDE_parameters(data_dicts, n)
    
    for r in tqdm(X):
        p_AL_lb, p_AL_ub, ci_lb, ci_ub = AL.compute_expected_profit_KDE(n, r, data_dicts, ub_v, v0, 
                                                          v_n1n,ghat_KDE_n1n,v_nn,ghat_KDE_nn,G_hat_vnn,
                                                          b_nns, b_n1ns,
                                                          integral_method=False, variedN=False, KDEs=None)
        profits_lb_AL.append(p_AL_lb)
        profits_ub_AL.append(p_AL_ub)
        ci_lbs.append(ci_lb)
        ci_ubs.append(ci_ub)
    return profits_lb_AL, profits_ub_AL, ci_lbs, ci_ubs

def compute_bounds_Bnn(X,n, data_dicts, ub_v, v0):
    profits_lb = []
    profits_ub = []
    
    b_nns, b_n1ns = get_bnns_bn1ns(data_dicts, n)
    
    # Get the KDE parameters
    v_n1n, _, v_nn, _, G_hat_vn1n, G_hat_vnn = get_KDE_parameters(data_dicts, n)
    
    for r in tqdm(X):
        p_lb, p_ub = AL_NEP.compute_expected_profit_KDE_Bnn(n, r, v0,
                                                            v_n1n, v_nn, G_hat_vn1n, G_hat_vnn,
                                                            b_nns, b_n1ns)
        profits_lb.append(p_lb)
        profits_ub.append(p_ub)
    
    return profits_lb, profits_ub

def compute_non_equilibrium_exact(X,n, data_dicts, ub_v, v0):
    profits = []
    
    _, _, v_nn, ghat_KDE_nn, _, G_hat_vnn = get_KDE_parameters(data_dicts, n)
    
    for r in X:
        p = NEP.compute_exp_profit_non_equil(n,r,data_dicts,v_nn,ghat_KDE_nn,G_hat_vnn,v0)
        profits.append(p)
    
    return profits

def compute_bounds_vN(X,n, data_dicts, ub_v, v0, KDEs):
    profits_lb_AL_vN = []
    profits_ub_AL_vN = []
    ci_lbs = []
    ci_ubs = []
    
    b_nns, b_n1ns = get_bnns_bn1ns(data_dicts, n)
    
    # Get the KDE parameters
    v_n1n, ghat_KDE_n1n, v_nn, ghat_KDE_nn, G_hat_vn1n, G_hat_vnn = get_KDE_parameters(data_dicts, n)
    
    for r in tqdm(X):
        p_AL_lb, p_AL_ub, ci_lb, ci_ub = AL.compute_expected_profit_KDE(n, r, data_dicts, ub_v, v0, 
                                                          v_n1n,ghat_KDE_n1n,v_nn,ghat_KDE_nn,G_hat_vnn,
                                                          b_nns, b_n1ns,
                                                          integral_method=False, variedN=True, KDEs=KDEs)
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
    X = np.linspace(0, UB_V, num_points)

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
    plt.ylim(-0.1, 1.)
    plt.plot(X,profits_lb_AL,color='tab:blue',linewidth=2, label='AL lb')    # marker=6 is a caretup
    plt.plot(X,profits_ub_AL,color='tab:blue',linewidth=2, label='AL ub')      # marker=7 is a caretdown
    plt.plot(X,ci_lbs,color='lightsteelblue', linestyle='dashed', label='95% CI lb')
    plt.plot(X,ci_ubs,color='lightsteelblue', linestyle='dashed', label='95% CI ub')
    
    # Plot the non-equilibrium exact
    profits_exact_non_equilibrium = compute_non_equilibrium_exact(X, n, data_dicts, UB_V, V0)
    plt.plot(X,profits_exact_non_equilibrium, color='black',linewidth=1,label='non-equilibrium')
    
    # Plot the non-equilibrium bounds
    profits_lb_non_equilibrium, profits_ub_non_equilibrium = compute_bounds_Bnn(X, n, data_dicts, UB_V, V0)
    plt.plot(X,profits_lb_non_equilibrium, color='tab:green',linewidth=2, label='lb (winning bid)')
    plt.plot(X,profits_ub_non_equilibrium, color='tab:green',linewidth=2, label='ub (winning bid)')
    
    plt.legend(loc='upper right')
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
