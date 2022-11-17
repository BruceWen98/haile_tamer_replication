import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

##### 1. Estimate Bounds of Profit #####

def vsOfF(results_dicts):
    # results_dict = [{'v':v, 'F_U':F_U}, ...] or [{'v':v, 'F_L':F_L}, ...]
    return [d['v'] for d in results_dicts]

def F_hat_U(p, results_dicts_F_U, vs_F_U):
    filtered_vs = [v for v in vs_F_U if v<=p]
    correct_v = max(filtered_vs)
    for d in results_dicts_F_U:
        if d['v']==correct_v:
            return d['F_U']

def F_hat_L(p, results_dicts_F_L, vs_F_L):
    filtered_vs = [v for v in vs_F_L if v>=p]
    correct_v = min(filtered_vs)
    for d in results_dicts_F_L:
        if d['v']==correct_v:
            return d['F_L']

def pi1_hat(p,v0, results_dicts_F_U, vs_F_U):
    return (p-v0)*(1-F_hat_U(p, results_dicts_F_U, vs_F_U))

def pi2_hat(p,v0, results_dicts_F_L, vs_F_L):
    return (p-v0)*(1-F_hat_L(p, results_dicts_F_L, vs_F_L))

def estimate_bounds(v0, results_dicts_F_U, vs_F_U, results_dicts_F_L, vs_F_L):
    # Check that vs_F_U and vs_F_L are the same
    if vs_F_U != vs_F_L:
        raise ValueError("vs_F_U and vs_F_L are not the same!")
    
    pi1s = []
    pi2s = []
    for p in vs_F_U:
        pi1 = pi1_hat(p,v0, results_dicts_F_U, vs_F_U)
        pi2 = pi2_hat(p,v0, results_dicts_F_L, vs_F_L)
        pi1s.append(pi1)
        pi2s.append(pi2)
    return pi1s, pi2s



F_U_dicts = pickle.load(open("F_U_dicts.pkl", "rb"))
F_L_dicts = pickle.load(open("F_L_dicts.pkl", "rb"))
vsOfF_U = vsOfF(F_U_dicts)
vsOfF_L = vsOfF(F_L_dicts)

# Estimate Bounds. Seller values good at 1 (relative price to auctioneer's estimate).
pi1s, pi2s = estimate_bounds(1, F_U_dicts, vsOfF_U, F_L_dicts, vsOfF_L)
sns.set_style('darkgrid')
plt.figure(figsize=(10,6), tight_layout=True)
plt.scatter(vsOfF_U,pi2s,color='orange', marker='|',label="high bound")
plt.scatter(vsOfF_U,pi1s,color='blue', marker='_', label="low bound")
plt.title("Bounds of Profit against Reserve (HT)")
plt.xlabel("Reserve")
plt.ylabel("Profit")


##### 2. Optimal Range of Reserve Prices, p_L to p_U #####

# Take maximum of pi1 as lower bound. Determine range of pi2 that are greater than this maximum.
def find_pL_pU(X, pi1s, pi2s):
    # X is the support of prices (x-axis)
    # pi1s is the corresponding values of pi1 (lower bound of profit function)
    # pi2s is the corresponding values of pi2 (upper bound of profit function)
    if len(pi1s) != len(pi2s):
        raise ValueError("pi1s and pi2s are not the same length!")
    pi1_star = max(pi1s)
    p1_star = vsOfF_U[pi1s.index(pi1_star)]
    print(p1_star)
    idx = np.where(pi2s <= pi1_star)[0]   # Find indices of elements less than pi1_star
    
    # pL
    possible_pL = [X[i] for i in idx if X[i]<p1_star]
    pL = max(possible_pL)
    #pU
    try:
        possible_pU = [X[i] for i in idx if X[i]>p1_star]
        pU = min(possible_pU)
    except ValueError:
        pU = p1_star
    return pL, pU
    

pL,pU = find_pL_pU(vsOfF_U, pi1s, pi2s)
plt.axvline(x=pL, color="blue", label="low optimal reserve bound")
plt.axvline(x=pU, color="red", label="high optimal reserve bound")
plt.legend()
plt.show()
