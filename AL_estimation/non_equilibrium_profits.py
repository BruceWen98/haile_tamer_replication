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

def find_nearest_below(arr, x):
    low = 0
    high = len(arr) - 1

    while low < high:
        mid = (low + high + 1) // 2
        if arr[mid] > x:
            high = mid - 1
        else:
            low = mid
    return low

def find_nearest_above(arr, x):
    low = 0
    high = len(arr) - 1

    while low < high:
        mid = (low + high) // 2
        if arr[mid] <= x:
            low = mid + 1
        else:
            high = mid
    if low == len(arr) - 1 and arr[low] <= x:  # x is larger than all elements
        return low
    return high  # otherwise return the index of the element just above x

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
            return b * g_hat_v[find_nearest_below(v_nn, b)]
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

def compute_exp_profit_non_equil(n,r,max_bid_dicts, v_nn, g_hat_v, G_hat_vnn, v0):
    term1 = calc_positive_term(max_bid_dicts, n, r, v_nn, g_hat_v)
    term2 = v0 * (1 - G_hat_vnn[find_nearest_below(v_nn, r)])
    return term1 - term2




# Kirill's Idea
# Bounded bounds on the profit
def inv_phi(n, phi):
    def f(s):
        return s**(n-2) * (1-s)    
    return n*(n-1) * integrate.quad(f, 0, phi)[0]

def Bnn_tilde_transform(n, b_nn, v_nn, G_hat_vnn):
    b1 = G_hat_vnn[find_nearest_below(v_nn, b_nn)] ** (1/n)
    b2 = inv_phi(n, b1)
    b_nn_tilde = v_nn[find_nearest_above(G_hat_vnn, b2)]
    return b_nn_tilde

def non_eq_LB(v_nn,G_hat_vnn,v_n1n,G_hat_vn1n,
              b_n1ns, r, v0):
    prob_bnn_GEQr = 1 - G_hat_vnn[find_nearest_below(v_nn,r)]
    prob_bn1n_GEQr = 1 - G_hat_vn1n[find_nearest_below(v_n1n,r)]

    # if bn1n>=r then bn1n, else 0
    exp_val = np.mean( [b_n1n if b_n1n>=r else 0 for b_n1n in b_n1ns] )

    return prob_bnn_GEQr*(r-v0) + exp_val - prob_bn1n_GEQr*r

def non_eq_UB(v_nn,G_hat_vnn,v_nn_tilde,G_hat_vnn_tilde,
              b_nns,b_nn_tildes,r,v0):    
    prob_bnn_GEQr = 1 - G_hat_vnn[find_nearest_below(v_nn,r)]
    prob_bnn_tilde_GEQr = 1 - G_hat_vnn_tilde[find_nearest_below(v_nn_tilde,r)]
    
    # if bnn_>=r then bnn_tilde, else 0
    exp_tilde = np.mean([b_nn_tilde if b_nn>=r else 0 for b_nn, b_nn_tilde in zip(b_nns, b_nn_tildes)])
    
    return prob_bnn_tilde_GEQr*(r-v0) + exp_tilde - prob_bnn_GEQr*r

def compute_exp_profit_PureNonEquilibrium_bounds(n,r,v0,max_bid_dicts, 
                                                 v_nn,G_hat_vnn,v_n1n,G_hat_vn1n,
                                                 v_nn_tilde,G_hat_vnn_tilde,
                                                 b_nns,b_n1ns,b_nn_tildes):
    b_nns = []
    b_n1ns = []
    for max_bid_dict in max_bid_dicts:
        if max_bid_dict['n'] == n:
            b_nn = max( max_bid_dict.get('1'), max_bid_dict.get('2') )
            b_n1n = min( max_bid_dict.get('1'), max_bid_dict.get('2') )
            b_nns.append(b_nn)
            b_n1ns.append(b_n1n)
    # b_nn_tildes = [Bnn_tilde_transform(n, b_nn, v_nn, G_hat_vnn) for b_nn in b_nns]
    
    lb = non_eq_LB(v_nn,G_hat_vnn,v_n1n,G_hat_vn1n,
              b_n1ns, r, v0)
    ub = non_eq_UB(v_nn,G_hat_vnn,v_nn_tilde,G_hat_vnn_tilde,
              b_nns,b_nn_tildes,r,v0)
    return lb, ub