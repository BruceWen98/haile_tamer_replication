import AL_profits as AL
import profit_diff as PD
import draw_AL_profits_specific as DAPS
import draw_non_equilibrium_profits as DNEP
import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

def compute_profit_diff_bounds(X,n,data_dicts,r0,v0):
    diff_lb = []
    diff_ub = []
    diff_lb_woCorrection = []
    diff_ub_woCorrection = []
    b_nns, b_n1ns = DAPS.get_bnns_bn1ns(data_dicts, n)
    
    _,_,v_nn,_,_,G_hat_vnn = DAPS.get_KDE_parameters(data_dicts, n)
    
    for r in tqdm(X):
        lb, ub = PD.compute_expected_profit_diff_r0(n,r,r0,v0,
                                                 v_nn, G_hat_vnn,
                                                 b_nns, b_n1ns, extraConstraint=True)
        lb_woCorrection, ub_woCorrection = PD.compute_expected_profit_diff_r0(n,r,r0,v0,
                                                 v_nn, G_hat_vnn,
                                                 b_nns, b_n1ns, extraConstraint=False)
        diff_lb.append(lb)
        diff_ub.append(ub)
        diff_lb_woCorrection.append(lb_woCorrection)
        diff_ub_woCorrection.append(ub_woCorrection)
    
    return diff_lb, diff_ub, diff_lb_woCorrection, diff_ub_woCorrection

def compute_exact_profit_diff(X,n,data_dicts,r0,v0):
    diffs = []
    
    _,_,v_nn,ghat_KDE_nn,_,G_hat_vnn = DAPS.get_KDE_parameters(data_dicts, n)
    
    for r in X:
        d = PD.exact_profit_diff(n,r,r0,data_dicts,
                                 v_nn, ghat_KDE_nn, G_hat_vnn,
                                 v0)
        diffs.append(d)
    
    return diffs

def compute_v0(dicts):
    ratios = []
    for d in dicts:
        ratios.append((d['low_rel_high']+1)/2)
    return np.median(ratios)

def compute_r0(dicts):
    ratios = []
    for d in dicts:
        ratios.append(d['low_rel_high'])
    return np.median(ratios)

def draw_profit_diff_specificN(INPATH, OUTPATH, n, ub_v=10, num_points=1000):
    ### 2. Input Data
    with open(INPATH, 'rb') as handle:
        data_file = pickle.load(handle)
        data_dicts = data_file['data']
        characteristics = data_file['characteristics']
        
    cat, loc, val, deg_competition = DAPS.getNamesForGraphs2(characteristics)
            
    print("Values of N = ", AL.calc_N_set(data_dicts))
    print("Size of this Data = ", len(data_dicts))

    v0 = compute_r0(data_dicts)
    r0 = compute_r0(data_dicts)
    X = np.linspace(0, ub_v, num_points)
    
    print("n = {}".format(n))
    print(len([d for d in data_dicts if d['n'] == n]))

    try:
        diff_lb, diff_ub, diff_lb_woCorrection, diff_ub_woCorrection = compute_profit_diff_bounds(X,n,data_dicts,r0,v0)
        diff = compute_exact_profit_diff(X,n,data_dicts,r0,v0)
    except ValueError as e:     #ValueError: Root finding did not converge. Need more data.
        print(e)
        return
    
    obs = len([d for d in data_dicts if d['n'] == n])
    sns.set_style('darkgrid')
    plt.figure(figsize=(10,6), tight_layout=True)
    plt.title("Category: {}, Loc: {}, Sell Price: {}, Competition: {};  N={}; Obs={}".format(cat,loc,val,deg_competition,n,obs), fontsize=13)
    plt.xlabel("Reserve Price")
    plt.ylabel(r'Expected Profit Difference relative to $r_0$')
    plt.ylim(-0.4, 0.4)
    plt.plot(X,diff,color='black',linewidth=2,linestyle='dashdot', label='non-equilibrium')
    plt.plot(X,diff_ub,color='tab:blue',linewidth=2, alpha=0.7, label=r'ub (LP sol w $F_{n:n}$ constraint)')
    plt.plot(X,diff_lb,color='tab:blue',linewidth=2, alpha=0.7, label=r'lb (LP sol w $F_{n:n}$ constraint)')
    plt.plot(X,diff_lb_woCorrection,color='tab:orange',linewidth=2, alpha=0.7, label='lb (LP sol)')
    plt.plot(X,diff_ub_woCorrection,color='tab:orange',linewidth=2, alpha=0.7, label='ub (LP sol)')
    plt.legend()
    
    # Print r0,v0
    plt.text(0, -0.35, r'$v_0={}$,$r_0={}$'.format(str(round(v0,2)),str(round(r0,2))) , 
             bbox=dict(facecolor='red', alpha=0.5))
    
    plt.savefig(OUTPATH + "profit_diff_n{}.png".format(n))
    return


