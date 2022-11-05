import numpy as np
import optimize as OPT
from tqdm import tqdm


## Estimation (Section C, Haile-Tamer 2003)

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


# The final estimation functions, for each v in [0,1].
def F_hat_U(max_bid_dicts,v,rho_T):
    vec_phis = []
    possible_ns = calc_M_set(max_bid_dicts)
    for n in possible_ns:
        for i in range(n-1,n+1):    # i = n-1,n only since we only have top 2 bids.
            phi = calc_eq10_phi(max_bid_dicts,i,n,v)
            vec_phis.append(phi)
    vec_phis = [phi for phi in vec_phis if phi == phi]  # remove nan
    vec_phis = [phi for phi in vec_phis if phi<=1.0 and phi>=0.0]  # remove values out of range
    return smooth_weighted_avg(vec_phis, rho_T)
    # return min(vec_phis)

def F_hat_L(max_bid_dicts,v,rho_T,increment):
    vec_phis = []
    possible_ns = calc_M_set(max_bid_dicts)
    for n in possible_ns: 
        if n==0 or n==1:
            continue
        phi = calc_eq11_phi(max_bid_dicts,n,v,increment)
        vec_phis.append(phi)
    vec_phis = [phi for phi in vec_phis if phi == phi]  # remove nan
    vec_phis = [phi for phi in vec_phis if phi<=1.0 and phi>=0.0]  # remove values out of range
    return smooth_weighted_avg(vec_phis, rho_T)
    # return max(vec_phis)
