import numpy as np
from tqdm import tqdm
import stieltjes
import math
import scipy.integrate as integrate
from scipy.optimize import minimize
from KDEpy import FFTKDE    # Kernel Density Estimation
import matplotlib.pyplot as plt
import AL_inference as ALinf

import warnings
warnings.filterwarnings("ignore")

########## CODE for Aradillas-Lopez 2013 ##########


##### 0. Numerical Integration of the pdf of the n-1:n order statistic. #####
def integrate_0_to_phi(n,phi):
    def f(s): 
        return (s**(n-2) * (1-s))
    return n * (n-1) * integrate.romberg(f,0,phi)

def calc_phi(n,H):
    bds = [(0,1)]
    # possibilities = [
    #     minimize(lambda phi: (integrate_0_to_phi(n,phi)-H)**2, x0=[0.7], method="Powell", bounds = bds).x[0],
    #     minimize(lambda phi: (integrate_0_to_phi(n,phi)-H)**2, x0=[0.7], method="Nelder-Mead", bounds = bds).x[0],
    #     minimize(lambda phi: (integrate_0_to_phi(n,phi)-H)**2, x0=[0.7], method="L-BFGS-B", bounds = bds).x[0],
    # ]
    return minimize(lambda phi: (integrate_0_to_phi(n,phi)-H)**2, x0=[0.7], method="Powell", bounds = bds).x[0]


##### 1. Supporting Functions #####

# upperbar{n} = max{num of bidders in each auction in the T auctions}
# This computes the possible n's.
def calc_N_set(max_bid_dicts):
    list_num_bidders = []
    for max_bid_dict in max_bid_dicts:
        num_bidders = max_bid_dict['n']
        list_num_bidders.append(num_bidders)
    return list(np.unique(list_num_bidders))

# Tn is the number of auctions (out of T) that have n bidders.
def calc_Tn(max_bid_dicts, n):
    count = 0
    for max_bid_dict in max_bid_dicts:
        if max_bid_dict['n'] == n:
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
        if max_bid_dict['n'] == n:
            # ith highest bid in the bid dictionary. Note that i are n-1, n.
            ith_bid = sorted([max_bid_dict.get(key) for key in ['1','2']], reverse=True)[n-i]
            Xi.append(ith_bid)
        continue
    
    # v, g_hat_v = FFTKDE(kernel=ker, bw=bandwidth).fit(Xi).evaluate()
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


##### 2. Estimation of bounds. (Valuations are Positively Dependent) #####
### 2c. Estimation of the F_{n:n} bounds using KDE instead. ###
def G_KDE(v, v_arr, G_hat_v):
    return G_hat_v[find_nearest(v_arr, v)]

def phiGnnN_KDE(n,v, v_arr, G_hat_v):
    return calc_phi(n, G_KDE(v,v_arr,G_hat_v)) ** n

def Fnn_KDE(n,v, v_arr, G_hat_v):
    return phiGnnN_KDE(n,v, v_arr, G_hat_v), G_KDE(v, v_arr, G_hat_v)


##### 3. Variations in N, the number of bidders. (pg. 498 Lemma 4, AL-Gandhi-Quint) ##### 
## 3b. Same as 3a, but using KDE instead.
def Gn1n_KDE(n, max_bid_dicts):
    v_arr, g_hat_v, h = ghat_KDE(max_bid_dicts, n-1, n)
    G_hat_v = KDE_pdf_to_cdf(v_arr, g_hat_v)
    return v_arr, G_hat_v, h

def getAll_KDEs(n, max_bid_dicts):  # Get all KDEs for m>=n+1
    Ns = calc_N_set(max_bid_dicts)
    KDEs = {}
    loop = [m for m in Ns if m >= n+1]  # m >= n+1, m<=N_bar
    for m in loop:
        m_v_arr, m_G_hat_v, _ = Gn1n_KDE(m, max_bid_dicts)
        KDEs[m] = (m_v_arr, m_G_hat_v)
    return KDEs

def get_v_Gn1n_KDE(m,v,KDEs):
    v_arr, G_hat_v = KDEs[m]
    return G_KDE(v, v_arr, G_hat_v)

def computeFm1ms_KDE(n, v, max_bid_dicts, KDEs):
    Ns = calc_N_set(max_bid_dicts)
    Fm1ms = {}
    loop = [m for m in Ns if m >= n+1]  # m >= n+1, m<=N_bar
    for m in loop:
        Fm1ms[m] = n/m/(m-1) * get_v_Gn1n_KDE(m,v,KDEs)
    return Fm1ms

def computeFnbar_KDE(v, max_bid_dicts, KDEs):
    Ns = calc_N_set(max_bid_dicts)
    return 1 / max(Ns) * get_v_Gn1n_KDE(max(Ns), v, KDEs) 

def computePhiFn1nbar_KDE(v, max_bid_dicts, KDEs):
    Ns = calc_N_set(max_bid_dicts)
    H = get_v_Gn1n_KDE(max(Ns), v, KDEs) 
    return 1/max(Ns) * ( calc_phi(max(Ns), H) ** max(Ns) )

def Fnn_variedN_KDE(n, v, max_bid_dicts, KDEs):
    Fm1ms = computeFm1ms_KDE(n, v, max_bid_dicts, KDEs)
    # Upper bound
    sum1 = sum([Fm1ms[m] for m in Fm1ms.keys() if m >= n+1])
    F_U = sum1 + n * computeFnbar_KDE(v, max_bid_dicts, KDEs)
    # Lower bound
    computePhiFn1nbar_KDE_val = computePhiFn1nbar_KDE(v, max_bid_dicts, KDEs)
    F_L = sum1 + n * computePhiFn1nbar_KDE_val
    return F_L,F_U, computePhiFn1nbar_KDE_val * max(calc_N_set(max_bid_dicts)), Fm1ms

##### 5. Estimation of the Expected Profit. #####
# Method: Using KDE to smooth the bounds
def compute_expected_profit_KDE(
    n, 
    r, 
    max_bid_dicts, 
    ub_v, 
    v0, 
    
    # KDE parameters (this is constant across any r)
    v_n1n,
    ghat_KDE_n1n,
    v_nn,
    ghat_KDE_nn,
    G_hat_vnn,
    
    # parameters for estimating without integrating
    b_nns,
    b_n1ns,
    
    # optional parameters
    integral_method=False,
    variedN=False, 
    KDEs=None
    ):

    if variedN==False:
        fnn_l, fnn_u = Fnn_KDE(n,r, v_nn, G_hat_vnn)
        L = len([d for d in max_bid_dicts if d['n'] == n])
    else:   # variedN==True
        ##### Original \bar{N} method in AL.(without intersecting bounds)
        fnn_l, fnn_u, phi, Fm1ms = Fnn_variedN_KDE(n,r, max_bid_dicts, KDEs)
        L = len([d for d in max_bid_dicts if d['n'] >= n])
    
    
    # if using integral method
    if integral_method==True:
        def f_ub(v):
            return max(r,v) * ghat_KDE_nn[find_nearest(v_nn, v)]
        def f_lb(v):
            return max(r,v) * ghat_KDE_n1n[find_nearest(v_n1n, v)]
        
        # 100 is an arbitrarily large number for the integral.
        term1_ub = integrate.quad(f_ub,0,85)[0]
        term1_lb = integrate.quad(f_lb,0,85)[0]
    else:   # use expectation method instead

        term1_ub = np.mean([max(r,b) for b in b_nns])
        term1_lb = np.mean([max(r,b) for b in b_n1ns])
        
    ub = term1_ub - v0 - fnn_l * (r - v0)
    lb = term1_lb - v0 - fnn_u * (r - v0)

    # inference
    if variedN==False:
        ci_lb, ci_ub = ALinf.CI_out(ub, lb, L, max_bid_dicts, n, r, v0, fnn_l, alpha=0.05)
    else: 
        ci_lb, ci_ub = ALinf.CI_out_varyingN(ub, lb, L, max_bid_dicts, n, r, v0, phi,Fm1ms, alpha=0.05)
    
    return lb,ub,ci_lb,ci_ub