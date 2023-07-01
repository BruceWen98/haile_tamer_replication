import numpy as np
import math
import scipy.integrate as integrate
from KDEpy import FFTKDE    # Kernel Density Estimation

def calc_N_set(max_bid_dicts):
    list_num_bidders = []
    for max_bid_dict in max_bid_dicts:
        num_bidders = max_bid_dict['n']
        list_num_bidders.append(num_bidders)
    return list(np.unique(list_num_bidders))

# estimate the positive term
def calc_positive_term(max_bid_dicts, n, r, v_nn, g_hat_v, integral_method = False):
    b_nns = []
    for max_bid_dict in max_bid_dicts:
        if max_bid_dict['n'] == n:
            b_nn = max( max_bid_dict.get('1'), max_bid_dict.get('2') )
            b_nns.append(b_nn)
    
    
    if integral_method == False:
        # if bnn>r then bnn, else 0
        b_nns = [b_nn if b_nn>r else 0 for b_nn in b_nns]
        return np.mean(b_nns)
    else:
        def f(b):
            return b * g_hat_v[find_nearest(v_nn, b)]
        return integrate.quad(f, r, np.inf)[0]
    
    

# For Kernel Density Estimation
def ghat_KDE(max_bid_dicts,i,n, ker='gaussian', bandwidth="ISJ"): # ISJ is the improved Sheather-Jones algorithm
    Xi = [] # This is b_{i:n}, the list of the ith highest bids in the Tn auctions.
    for max_bid_dict in max_bid_dicts:
        if max_bid_dict['n'] == n:
            # ith highest bid in the bid dictionary. Note that i are n-1, n.
            ith_bid = sorted([max_bid_dict.get(key) for key in ['1','2']], reverse=True)[n-i]
            Xi.append(ith_bid)
        continue
    
    kde = FFTKDE(kernel=ker, bw=bandwidth)
    v, g_hat_v = kde.fit(Xi).evaluate()
    h = kde.bw
    return v, g_hat_v, h

def KDE_pdf_to_cdf(v, g_hat_v):
    sum_g_hat_v = np.sum(g_hat_v)
    G_hat_v = np.zeros(len(v))
    for k in range(len(v)):
        G_hat_v[k] = np.sum(g_hat_v[:k])/sum_g_hat_v
    return G_hat_v

def find_nearest(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx-1
    else:
        return idx

def compute_exp_profit_non_equil(n,r,max_bid_dicts, v_nn, g_hat_v, G_hat_vnn, v0):
    term1 = calc_positive_term(max_bid_dicts, n, r, v_nn, g_hat_v)
    term2 = v0 * (1 - G_hat_vnn[find_nearest(v_nn, r)])
    return term1 - term2


    
    