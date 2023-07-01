import numpy as np
import optimize as OPT
from tqdm import tqdm
from scipy.optimize import minimize
import scipy.integrate as integrate

##### HELPER FUNCTIONS #####

# upperbar{M} = max{num of bidders in each auction in the T auctions}
# This is the theoretical upper bound on the number of bidders in the T auctions involved.
def calc_M_set(max_bid_dicts):
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
# \hat{G}_{i:n}(v) is the empirical distribution function used to calculate upper bound.
def calc_Ghat(max_bid_dicts,i,n,v):
    Tn = calc_Tn(max_bid_dicts,n)
    count = 0
    for max_bid_dict in max_bid_dicts:
        if max_bid_dict['n'] == n:
            # ith smallest bid in the bid dictionary. Note that i are 1,2.
            ith_bid = sorted([max_bid_dict.get(key) for key in ['1','2']], reverse=True)[n-i]
            if ith_bid <= v:
                count+=1
        continue
    return count/Tn

# Equation 9, Haile-Tamer 2003
# \hat{G}_{n:n}^{\delta}(v) is the empirical distribution function used to calculate lower bound.
def calc_Ghat_inc(max_bid_dicts,n,v,increment):
    Tn = calc_Tn(max_bid_dicts,n)
    count = 0
    for max_bid_dict in max_bid_dicts:
        if max_bid_dict['n'] == n:
            # max bid in max_bid_dict.
            nt_bid = max([max_bid_dict.get(key) for key in ['1','2']])
            if nt_bid*(1+increment)<= v:  ## Assume that increment is same across all the auctions.
                count+=1
        continue
    return count/Tn


# Equation 10, Haile-Tamer 2003  (but not the min - later we will use the softmax, eq. 12)
def calc_eq10_phi(max_bid_dicts,i,n,v):
    H = calc_Ghat(max_bid_dicts,i,n,v)
    phi = OPT.calc_phi(i,n,H)
    return phi

# Equation 11, Haile-Tamer 2003  (but not the max - later we will use the softmax)
def calc_eq11_phi(max_bid_dicts,n,v,increment):
    H = calc_Ghat_inc(max_bid_dicts,n,v,increment)
    phi = OPT.calc_phi(n-1,n,H)
    return phi


# Equation 12, Haile-Tamer 2003
def smooth_weighted_avg(vec_phis, rho_T):
    J = len(vec_phis)

    # The Common Denominator across all j in J.
    denom = 0
    for k in range(J):
        denom += np.exp(vec_phis[k]*rho_T)

    sum = 0
    for j in range(J):
        y_hat_j = vec_phis[j]
        sum += y_hat_j * np.exp(y_hat_j*rho_T)/denom

    return sum


#Numerical Integration of the pdf of the n-1:n order statistic.
def integrate_0_to_phi(n,phi):
    def f(s): 
        return (s**(n-2) * (1-s))
    return n * (n-1) * integrate.quad(f,0,phi)[0]

#Numerical Integration of the pdf of the n:n order statistic.
def integrate_0_to_phi_highest(n,phi):
    def f(s): 
        return s**(n-1)
    return n * integrate.quad(f,0,phi)[0]

def calc_phi(n,H):
    bds = [(0,1)]
    return minimize(lambda phi: (integrate_0_to_phi(n,phi)-H)**2, x0=[0.7], method="Powell", bounds = bds).x[0]

def calc_phiNN(n,H):
    bds = [(0,1)]
    return minimize(lambda phi: (integrate_0_to_phi_highest(n,phi)-H)**2, x0=[0.7], method="Powell", bounds = bds).x[0]

def phi_inv_n1n(n, Fb):
    return integrate_0_to_phi(n,Fb)

def phi_inv_nn(n, Fb):
    return integrate_0_to_phi_highest(n,Fb)

##### END OF HELPER FUNCTIONS #####


## AL-Bounds 
# a. Estimation of the F_{n-1:n} bounds. ###
def Gnn(n, v, max_bid_dicts):
    return calc_Ghat(max_bid_dicts, n, n, v)

def Gn1n(n, v, max_bid_dicts):
    return calc_Ghat(max_bid_dicts, n-1, n, v)

def Fn1n(n, v, max_bid_dicts):
    return Gnn(n, v, max_bid_dicts), Gn1n(n, v, max_bid_dicts)

# b. Estimation of the F_{n:n} bounds. ###
def phiGnnN(n, v, max_bid_dicts):
    return calc_phi(n, Gnn(n, v, max_bid_dicts)) ** n

def Fnn(n, v, max_bid_dicts):
    return phiGnnN(n, v, max_bid_dicts), Gnn(n, v, max_bid_dicts)




## Haile-Tamer Bounds
# The final estimation functions, for each v in [0,1].
def F_HT_n1n(n,max_bid_dicts,v):
    vec_phis = []
    for i in range(n-1,n+1):    # i = n-1,n only since we only have top 2 bids.
        phi = calc_eq10_phi(max_bid_dicts,i,n,v)
        vec_phis.append(phi)
    # return Gnn(n,v,max_bid_dicts),phi_inv_n1n(n,smooth_weighted_avg(vec_phis, rho_T))
    return Gnn(n,v,max_bid_dicts),phi_inv_n1n(n,min(vec_phis))

def F_HT_nn(n,max_bid_dicts,v):
    vec_phis = []
    for i in range(n-1,n+1):    # i = n-1,n only since we only have top 2 bids.
        phi = calc_eq10_phi(max_bid_dicts,i,n,v)
        vec_phis.append(phi)
    # return calc_phi(n, Gnn(n, v, max_bid_dicts))**n, phi_inv_nn(n,smooth_weighted_avg(vec_phis, rho_T))
    return calc_phi(n, Gnn(n, v, max_bid_dicts))**n, phi_inv_nn(n,min(vec_phis))