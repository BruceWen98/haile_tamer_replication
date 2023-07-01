import numpy as np
from tqdm import tqdm
import math
import scipy.integrate as integrate
from scipy.optimize import minimize
from KDEpy import FFTKDE    # Kernel Density Estimation
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")


##### 0. Numerical Integration of the pdf of the n-1:n order statistic. #####
def integrate_0_to_phi(n,phi):
    def f(s): 
        return (s**(n-2) * (1-s))
    return n * (n-1) * integrate.romberg(f,0,phi)[0]

def calc_phi(n,H):
    bds = [(0,1)]
    return minimize(lambda phi: (integrate_0_to_phi(n,phi)-H)**2, x0=[0.7], method="Powell", bounds = bds).x[0]


##### 1. Supporting Functions #####
# This computes the possible n's.
def calc_N_set(max_bid_dicts):
    list_num_bidders = []
    for max_bid_dict in max_bid_dicts:
        num_bidders = max_bid_dict['n']
        list_num_bidders.append(num_bidders)
    return list(np.unique(list_num_bidders))

# Tn is the number of auctions (out of T) that have at least n bidders.
def calc_Tn(max_bid_dicts, n):
    count = 0
    for max_bid_dict in max_bid_dicts:
        if max_bid_dict['n'] >= n:
            count+=1
    return count

# Search for the nearest idx(value) in a sorted array, i.e. v_n1n or v_nn. This is to the left of the value.
def find_nearest(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx-1
    else:
        return idx
    
# For Kernel Density Estimation
def ghat_KDE(max_bid_dicts,i,n, ker='gaussian', bandwidth="ISJ"): # ISJ is the improved Sheather-Jones algorithm
    Xi = [] # This is b_{i:n}, the list of the ith highest bids in the Tn auctions.
    for max_bid_dict in max_bid_dicts:
        if max_bid_dict['n'] >= n:
            # ith highest bid in the bid dictionary. Note that i are n-1, n.
            ith_bid = sorted([max_bid_dict.get(key) for key in ['1','2']], reverse=True)[n-i]
            Xi.append(ith_bid)
        continue
    
    v, g_hat_v = FFTKDE(kernel=ker, bw=bandwidth).fit(Xi).evaluate()
    return v, g_hat_v

def KDE_pdf_to_cdf(v, g_hat_v):
    sum_g_hat_v = np.sum(g_hat_v)
    G_hat_v = np.zeros(len(v))
    for k in range(len(v)):
        G_hat_v[k] = np.sum(g_hat_v[:k])/sum_g_hat_v
    return G_hat_v

##### 2. Estimation of bounds. (Valuations are Positively Dependent) #####
def G_KDE(v, v_arr, G_hat_v):
    return G_hat_v[find_nearest(v_arr, v)]

def phiGnnN_KDE(n,v, v_arr, G_hat_v):
    return calc_phi(n, G_KDE(v,v_arr,G_hat_v)) ** n

def Fnn_KDE(n,v, v_arr, G_hat_v):
    return phiGnnN_KDE(n,v, v_arr, G_hat_v), G_KDE(v, v_arr, G_hat_v)

##### 4. Estimation of the Expected Profit. #####
def compute_expected_profit_KDE(n, r, max_bid_dicts, ub_v, v0, variedN=False, KDEs=None):
    v_n1n, ghat_KDE_n1n = ghat_KDE(max_bid_dicts,n-1,n, ker='gaussian', bandwidth="ISJ")
    v_nn, ghat_KDE_nn = ghat_KDE(max_bid_dicts,n,n, ker='gaussian', bandwidth="ISJ")

    G_hat_vnn = KDE_pdf_to_cdf(v_nn, ghat_KDE_nn)
    fnn_l, fnn_u = Fnn_KDE(n,r, v_nn, G_hat_vnn)
        
    def f_ub(v):
        return max(r,v) * ghat_KDE_nn[find_nearest(v_nn, v)]
    def f_lb(v):
        return max(r,v) * ghat_KDE_n1n[find_nearest(v_n1n, v)]
    
    # 100 is an arbitrarily large number for the integral.
    ub = integrate.quad(f_ub,0,85)[0] - v0 - fnn_l * (r - v0)
    lb = integrate.quad(f_lb,0,85)[0] - v0 - fnn_u * (r - v0)

    return lb,ub