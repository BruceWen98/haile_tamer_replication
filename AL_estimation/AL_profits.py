import numpy as np
from tqdm import tqdm
import stieltjes
import scipy.integrate as integrate
from scipy.optimize import minimize

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

# Equation 8, Haile-Tamer 2003
## \hat{G}_{i:n}(v) is the empirical distribution function (CDF).
## Here, i=n-1 or i=n.
def calc_Ghat(max_bid_dicts,i,n,v):
    Tn = calc_Tn(max_bid_dicts,n)
    count = 0
    for max_bid_dict in max_bid_dicts:
        if max_bid_dict['n'] == n:
            # ith smallest bid in the bid dictionary. Note that i are n-1, n.
            ith_bid = sorted([max_bid_dict.get(key) for key in ['1','2']], reverse=True)[n-i]
            if ith_bid <= v:
                count+=1
        continue
    return count/Tn


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



##### 3. Variations in N, the number of bidders. (pg. 498 Lemma 4, AL) ##### 

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



##### 4. Estimation of the Expected Profit. #####
def compute_expected_profit(n, r, max_bid_dicts, ub_v, v0, h, variedN=False):
    si = stieltjes.stieltjes_integral(lambda v: max(r,v), lambda v: Fn1n(n,v,max_bid_dicts)[0], 0, ub_v, 10, h)
    
    if variedN==False:
        fnn_l, fnn_u = Fnn(n, r, max_bid_dicts)
    else:
        fnn_l, fnn_u = Fnn_variedN(n, r, max_bid_dicts)
        
    print("fnn_l: ", fnn_l)
    print("fnn_u: ", fnn_u)
    lb = si - v0 - fnn_u * (r - v0)
    ub = si - v0 - fnn_l * (r - v0)
    return lb,ub