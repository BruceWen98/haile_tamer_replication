# In this file, we will compute the expected profit difference between:
# i) Setting reserve as v0
# ii) Setting reserve as some r other than v0

import numpy as np
import AL_profits as AL
import non_equilibrium_profits as NEP

def compute_expected_profit_diff(n,r,v0,
                                 v_nn,G_hat_vnn,
                                 b_nns, b_n1ns,
                                ):
    fnn_l, fnn_u = AL.Fnn_KDE(n,r, v_nn, G_hat_vnn)
    
    i1 = np.mean([max(r,b) for b in b_nns])
    i2 = np.mean([max(v0,b) for b in b_n1ns])
    i3 = np.mean([max(r,b) for b in b_n1ns])
    i4 = np.mean([max(v0,b) for b in b_nns])
    
    ub = (i1-i2) - fnn_l*(r-v0)
    lb = (i3-i4) - fnn_u*(r-v0)
    return lb, ub

def compute_expected_profit_diff_withCorrection(n,r,v0,
                                                v_nn,G_hat_vnn,
                                                b_nns, b_n1ns,
                                                ):
    fnn_l, fnn_u = AL.Fnn_KDE(n,r, v_nn, G_hat_vnn)
    
    i1 = np.mean([max(r,b) for b in b_nns])
    i2 = np.mean([max(v0,b) for b in b_n1ns])
    i3 = np.mean([max(r,b) for b in b_n1ns])
    i4 = np.mean([max(v0,b) for b in b_nns])
    
    if r>v0:    # correction for ub
        ub = (i1-i2) - max(fnn_l, AL.G_KDE(v0,v_nn,G_hat_vnn)) * (r-v0)
    else:
        ub = (i1-i2) - fnn_l*(r-v0)
    lb = (i3-i4) - fnn_u*(r-v0)
    
    return lb, ub

def compute_expected_profit_diff_r0(n,r,r0,v0,
                                    v_nn,G_hat_vnn,
                                    b_nns, b_n1ns,
                                    ):
    fnn_l, fnn_u = AL.Fnn_KDE(n,r, v_nn, G_hat_vnn)
    i1 = np.mean([max(r,b) for b in b_nns])
    i2 = np.mean([max(r0,b) for b in b_n1ns])
    i3 = np.mean([max(r,b) for b in b_n1ns])
    i4 = np.mean([max(r0,b) for b in b_nns])
    

def exact_profit_diff(n,r,max_bid_dicts, v_nn, g_hat_v, G_hat_vnn, v0):
    exp_bnn_geq_r = NEP.calc_positive_term(max_bid_dicts, n, r, v_nn, g_hat_v)
    exp_bnn_geq_v0 = NEP.calc_positive_term(max_bid_dicts, n, v0, v_nn, g_hat_v)
    term1 = exp_bnn_geq_r - exp_bnn_geq_v0
    
    term2 = v0 * (G_hat_vnn[NEP.find_nearest(v_nn, v0)]
                  - G_hat_vnn[NEP.find_nearest(v_nn, r)])
    
    return term1 - term2
