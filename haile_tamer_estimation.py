import numpy as np
import numerical_ODE as NODE
from tqdm import tqdm


## Estimation (Section C, Haile-Tamer 2003)

# upperbar{M} = max{num of bids in each auction in the T auctions}
# This is the theoretical upper bound on the number of bidders in the T auctions involved.
def calc_M(all_bids):
    list_num_bids = []
    for list_bids in all_bids:
        num_bids = len(list_bids)
        list_num_bids.append(num_bids)
    return max(list_num_bids)

# Tn is the number of auctions (out of T) that have n bids.
def calc_Tn(all_bids, n):
    count = 0
    for list_bids in all_bids:
        if len(list_bids) == n:
            count+=1
    return count

# Equation 8, Haile-Tamer 2003
# \hat{G}_{i:n}(v) is the empirical distribution function used to calculate upper bound.
def calc_Ghat(all_bids,i,n,v):
    Tn = calc_Tn(all_bids,n)
    count = 0
    for list_bids in all_bids:
        if len(list_bids) == n:
            # ith largest bid in list_bids. Note that i starts from 1.
            ith_bid = list(reversed(sorted(list_bids)))[i-1]
            if ith_bid <= v:
                count+=1
    return count/Tn

# Equation 9, Haile-Tamer 2003
# \hat{G}_{n:n}^{\delta}(v) is the empirical distribution function used to calculate lower bound.
def calc_Ghat_inc(all_bids,i,n,v,increment):
    Tn = calc_Tn(all_bids,n)
    count = 0
    for list_bids in all_bids:
        if len(list_bids) == n:
            # lowest bid in list_bids.
            nt_bid = min(list_bids)
            if nt_bid*(1+increment)<= v:  ## Assume that increment is same across all the auctions.
                count+=1
    return count/Tn


# Equation 10, Haile-Tamer 2003  (but not the min - later we will use the softmax, eq. 12)
def calc_eq10_phi(all_bids,i,n,v):
    H = calc_Ghat(all_bids,i,n,v)
    phi = NODE.newton_method_repeated(i,n,H, max_iters=100,tries=20)
    return phi

# Equation 11, Haile-Tamer 2003  (but not the max - later we will use the softmax)
def calc_eq11_phi(all_bids,i,n,v,increment):
    H = calc_Ghat_inc(all_bids,i,n,v,increment)
    phi = NODE.newton_method_repeated(n-1,n,H, max_iters=100,tries=20)
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
def F_hat_U(all_bids,M,v,rho_T):
    vec_phis = []
    possible_ns = list(np.unique(list(map(len, all_bids))))
    for n in possible_ns: 
        for i in range(1,n+1):
            phi = calc_eq10_phi(all_bids,i,n,v)
            vec_phis.append(phi)
    vec_phis = [phi for phi in vec_phis if phi == phi]  # remove nan
    vec_phis = [phi for phi in vec_phis if phi<=1.0 and phi>=0.0]  # remove values out of range
    return smooth_weighted_avg(vec_phis, rho_T)

def F_hat_L(all_bids,M,v,rho_T,increment):
    vec_phis = []
    possible_ns = list(np.unique(list(map(len, all_bids))))
    for n in possible_ns: 
        if n==0:
            continue
        for i in range(1,n+1):
            phi = calc_eq11_phi(all_bids,i,n,v,increment)
            vec_phis.append(phi)
    vec_phis = [phi for phi in vec_phis if phi == phi]  # remove nan
    vec_phis = [phi for phi in vec_phis if phi<=1.0 and phi>=0.0]  # remove values out of range
    return smooth_weighted_avg(vec_phis, rho_T)
