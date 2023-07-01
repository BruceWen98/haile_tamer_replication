import numpy as np
from tqdm import tqdm
import math
import scipy.integrate as integrate
from scipy.optimize import minimize
from KDEpy import FFTKDE    # Kernel Density Estimation
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

########## CODE for Aradillas-Lopez 2013 ##########


##### 0. Numerical Integration of the pdf of the n-1:n order statistic. #####
def integrate_0_to_phi(n,phi):
    def f(s): 
        return (s**(n-2) * (1-s))
    return n * (n-1) * integrate.quad(f,0,phi)[0]

def calc_phi(n,H):
    bds = [(0,1)]
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

# Equation 8, Haile-Tamer 2003
## \hat{G}_{i:n}(v) is the empirical distribution function (CDF).
## Here, i=n-1 or i=n.
def calc_Ghat(max_bid_dicts,i,n,v):
    Tn = calc_Tn(max_bid_dicts,n)
    count = 0
    for max_bid_dict in max_bid_dicts:
        if max_bid_dict['n'] == n:
            # ith highest bid in the bid dictionary. Note that i are n-1, n.
            ith_bid = sorted([max_bid_dict.get(key) for key in ['1','2']], reverse=True)[n-i]
            if ith_bid <= v:
                count+=1
        continue
    return count/Tn


# For Kernel Density Estimation
def ghat_KDE(max_bid_dicts,i,n, ker='gaussian', bandwidth="ISJ"): # ISJ is the improved Sheather-Jones algorithm
    Xi = [] # This is b_{i:n}, the list of the ith highest bids in the Tn auctions.
    for max_bid_dict in max_bid_dicts:
        if max_bid_dict['n'] == n:
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
### 2a. Estimation of the F_{n-1:n} bounds. ###
def Gnn(n, v, max_bid_dicts):
    return calc_Ghat(max_bid_dicts, n, n, v)

def Gn1n(n, v, max_bid_dicts):
    return calc_Ghat(max_bid_dicts, n-1, n, v)

def Fn1n(n, v, max_bid_dicts):
    return Gnn(n, v, max_bid_dicts), Gn1n(n, v, max_bid_dicts)

### 2b. Estimation of the F_{n:n} bounds. ###
def phiGnnN(n, v, max_bid_dicts):
    return calc_phi(n, Gnn(n, v, max_bid_dicts)) ** n

def Fnn(n, v, max_bid_dicts):
    return phiGnnN(n, v, max_bid_dicts), Gnn(n, v, max_bid_dicts)

### 2c. Estimation of the F_{n:n} bounds using KDE instead. ###
def G_KDE(v, v_arr, G_hat_v):
    return G_hat_v[find_nearest(v_arr, v)]

def phiGnnN_KDE(n,v, v_arr, G_hat_v):
    return calc_phi(n, G_KDE(v,v_arr,G_hat_v)) ** n

def Fnn_KDE(n,v, v_arr, G_hat_v):
    return phiGnnN_KDE(n,v, v_arr, G_hat_v), G_KDE(v, v_arr, G_hat_v)



##### 3. Variations in N, the number of bidders. (pg. 498 Lemma 4, AL-Gandhi-Quint) ##### 

## 3a. Non-KDE Method. ##
def computeFm1ms(n, v, max_bid_dicts):
    Ns = calc_N_set(max_bid_dicts)
    Fm1ms = {}
    loop = [m for m in Ns if m >= n+1]  # m >= n+1, m<=N_bar
    for m in loop:
        Fm1ms[m] = n/m/(m-1) * Gn1n(m, v, max_bid_dicts)
    return Fm1ms

def computeFnbar(v, max_bid_dicts):
    Ns = calc_N_set(max_bid_dicts)
    return 1 / max(Ns) * Gn1n(max(Ns), v, max_bid_dicts)    # Note: No n involved.

def computePhiFn1nbar(v, max_bid_dicts):
    Ns = calc_N_set(max_bid_dicts)
    H = Gn1n(max(Ns), v, max_bid_dicts)
    return 1/max(Ns) * ( calc_phi(max(Ns), H) ** max(Ns) )  # Note: No n involved.

def Fnn_variedN(n, v, max_bid_dicts):
    Fm1ms = computeFm1ms(n, v, max_bid_dicts)
    # Upper bound
    sum1 = sum([Fm1ms[m] for m in Fm1ms.keys() if m >= n+1])
    F_U =  sum1 + n * computeFnbar(v, max_bid_dicts)
    # Lower bound
    F_L = sum1 + n * computePhiFn1nbar(v, max_bid_dicts)
    return F_L,F_U


## 3b. Same as 3a, but using KDE instead.
def Gn1n_KDE(n, max_bid_dicts):
    v_arr, g_hat_v = ghat_KDE(max_bid_dicts, n-1, n)
    G_hat_v = KDE_pdf_to_cdf(v_arr, g_hat_v)
    return v_arr, G_hat_v

def getAll_KDEs(n, max_bid_dicts):  # Get all KDEs for m>=n+1
    Ns = calc_N_set(max_bid_dicts)
    KDEs = {}
    loop = [m for m in Ns if m >= n+1]  # m >= n+1, m<=N_bar
    for m in loop:
        m_v_arr, m_G_hat_v = Gn1n_KDE(m, max_bid_dicts)
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
    F_L = sum1 + n * computePhiFn1nbar_KDE(v, max_bid_dicts, KDEs)
    return F_L,F_U

def computeFntop_KDE(v, ntop, KDEs):
    return 1 / ntop * get_v_Gn1n_KDE(ntop, v, KDEs) 

def computePhiFn1ntop_KDE(v, ntop, KDEs):
    H = get_v_Gn1n_KDE(ntop, v, KDEs) 
    return 1/ntop * ( calc_phi(ntop, H) ** ntop )

def Fnn_variedN_KDE_tight(n, v, max_bid_dicts, KDEs):
    Fm1ms = computeFm1ms_KDE(n, v, max_bid_dicts, KDEs)
    
    ns = [m for m in Fm1ms.keys() if m >= n+1]
    F_Us = []
    F_Ls = []
    for n_top in ns:
        # Upper bound
        sum1 = sum([Fm1ms[m] for m in Fm1ms.keys() if m >= n+1 and m <= n_top])
        f_U =  sum1 + n * computeFntop_KDE(v, n_top, KDEs)
        # Lower bound
        f_L =  sum1 + n * computePhiFn1ntop_KDE(v, n_top, KDEs)
        F_Us.append(f_U)
        F_Ls.append(f_L)

    return F_Us, F_Ls

##### 4. Calculate Expected Revenue using Law of Large Numbers (LLN). #####
def draw_uniform01(N):
    return np.random.uniform(0, 1, N)

def computeExp(r,N,Fn1n_inv):
    drawsUniform01 = draw_uniform01(N)
    f = lambda x: Fn1n_inv(x)
    vf = np.vectorize(f)
    v_i = vf(drawsUniform01)
    return 1/N * np.sum(np.maximum(np.repeat(r, N), v_i))

def closestXidx(x, arr):
    return np.abs(arr - x).argmin()

##### 5. Estimation of the Expected Profit. #####
# Method: using LLN method (expected value)
def compute_expected_profit_LLN(n, r, max_bid_dicts, ub_v, v0, variedN=False):
    f = lambda v: Fn1n(n, v, max_bid_dicts)
    vf = np.vectorize(f)
    V_support = np.linspace(0, ub_v, 1000)

    Gnn,Gn1n = vf(V_support)
    
    exp_rev_ub = computeExp(r, 100000, lambda x: V_support[closestXidx(x,Gnn)])     # Number of draws for LLN = 100,000
    exp_rev_lb = computeExp(r, 100000, lambda x: V_support[closestXidx(x,Gn1n)])
    
    if variedN==False:
        fnn_l, fnn_u = Fnn(n, r, max_bid_dicts)
    else:
        fnn_l, fnn_u = Fnn_variedN(n, r, max_bid_dicts)
        
    # print("fnn_l: ", fnn_l)
    # print("fnn_u: ", fnn_u)
    lb = exp_rev_lb - v0 - fnn_u * (r - v0)
    ub = exp_rev_ub - v0 - fnn_l * (r - v0)
    print("Expected Profit bounds: ", lb, ub)
    return lb,ub

# Method: New version of LLN, without using the inverse function.
def compute_expected_profit_LLN_2(n, r, max_bid_dicts, ub_v, v0, variedN=False):
    max_bid_dicts = [d for d in max_bid_dicts if d['1']==d['1'] and d['2']==d['2']]  # remove NaNs
    T = np.sum([d['n']==n for d in max_bid_dicts])

    exp_rev_ub = 1/T * sum( np.maximum( np.repeat(r, T), [d['1'] for d in max_bid_dicts if d['n']==n] ) )
    exp_rev_lb = 1/T * sum( np.maximum( np.repeat(r, T), [d['2'] for d in max_bid_dicts if d['n']==n] ) ) 

    if variedN==False:
        fnn_l, fnn_u = Fnn(n, r, max_bid_dicts)
    else:
        fnn_l, fnn_u = Fnn_variedN(n, r, max_bid_dicts)

    lb = exp_rev_lb - v0 - fnn_u * (r - v0)
    ub = exp_rev_ub - v0 - fnn_l * (r - v0)
    
    if (lb>ub):
        print("lb>ub", r, lb, ub)
        print(fnn_l, fnn_u)
        print(exp_rev_lb, exp_rev_ub)

    return lb,ub

# Method: Using KDE to smooth the bounds
def compute_expected_profit_KDE(n, r, max_bid_dicts, ub_v, v0, variedN=False, KDEs=None):
    v_n1n, ghat_KDE_n1n = ghat_KDE(max_bid_dicts,n-1,n, ker='gaussian', bandwidth="ISJ")
    v_nn, ghat_KDE_nn = ghat_KDE(max_bid_dicts,n,n, ker='gaussian', bandwidth="ISJ")

    G_hat_vnn = KDE_pdf_to_cdf(v_nn, ghat_KDE_nn)

    if variedN==False:
        fnn_l, fnn_u = Fnn_KDE(n,r, v_nn, G_hat_vnn)
    else:   # variedN==True
        ##### Original \bar{N} method in AL.(without intersecting bounds)
        fnn_l, fnn_u = Fnn_variedN_KDE(n,r, max_bid_dicts, KDEs)
        
        # ##### Intersecting bounds Method.
        # F_Us, F_Ls = Fnn_variedN_KDE_tight(n,r, max_bid_dicts, KDEs)
        # orig_fnn_l, orig_fnn_u = Fnn_KDE(n,r, v_nn, G_hat_vnn)
        # # Union all of the fnn bounds
        # F_Us.append(orig_fnn_u)
        # F_Ls.append(orig_fnn_l)
        # # Intersect the bounds, criterion is max the lower bound,
        # # and min the upper bound such that it is still above lower bound.
        # fnn_l = max(F_Ls)
        # fnn_u = min(F_Us)
        # # try:
        # #     fnn_u = min([x for x in F_Us if x>=fnn_l and x<=orig_fnn_u])
        # # except:
        # #     fnn_u = orig_fnn_u
        # # fnn_u = F_Us[F_Ls.index(fnn_l)]
        
        
    def f_ub(v):
        return max(r,v) * ghat_KDE_nn[find_nearest(v_nn, v)]
    def f_lb(v):
        return max(r,v) * ghat_KDE_n1n[find_nearest(v_n1n, v)]
    
    # 100 is an arbitrarily large number for the integral.
    ub = integrate.quad(f_ub,0,100)[0] - v0 - fnn_l * (r - v0)
    lb = integrate.quad(f_lb,0,100)[0] - v0 - fnn_u * (r - v0)

    return lb,ub