import AL_profits as AL
import numpy as np

##### Estimation of the Expected Profit. #####
# Here, we use B_{n:n} rather than V_{n-1:n} to compute the profit.

def compute_expected_profit_KDE_Bnn(
    n, 
    r, 
    v0, 
    
    # KDE parameters (this is constant across any r)
    v_n1n,
    v_nn,
    G_hat_vn1n,
    G_hat_vnn,
    
    # parameters for estimating without integrating
    b_nns,
    b_n1ns,
    ):

    fnn_l, fnn_u = AL.Fnn_KDE(n,r, v_nn, G_hat_vnn)
    # L = len([d for d in max_bid_dicts if d['n'] == n])

    term1_ub = np.mean([b_nn     if b_nn>r else 0   for b_nn in b_nns])
    term1_lb = np.mean([b_nns[i] if b_n1n>r else 0  for i, b_n1n in enumerate(b_n1ns) ])
    
    ub = term1_ub - v0 + r*G_hat_vn1n[AL.find_nearest(v_n1n,r)] - fnn_l * (r - v0)
    lb = term1_lb - v0 + v0*fnn_u
    
    return lb,ub