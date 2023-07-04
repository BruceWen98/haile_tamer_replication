# In this file, we will compute the expected profit difference between:
# i) Setting reserve as v0
# ii) Setting reserve as some r other than v0

import numpy as np
import AL_profits as AL

def compute_expected_profit_diff(n,r,max_bid_dicts,v0,
                                 v_n1n,ghat_KDE_n1n,v_nn,ghat_KDE_nn,G_hat_vnn,
                                 b_nns, b_n1ns,
                                 KDEs=None 
                                ):
    
    fnn_l, fnn_u = AL.Fnn_KDE(n,r, v_nn, G_hat_vnn)
    term1_ub = np.mean([max(r,b) - max(v0,b) for b in b_nns])
    term1_lb = np.mean([max(r,b) - max(v0,b) for b in b_n1ns])
    
    ub = term1_ub - fnn_l * (r - v0)
    lb = term1_lb - fnn_u * (r - v0)
    
    return lb, ub