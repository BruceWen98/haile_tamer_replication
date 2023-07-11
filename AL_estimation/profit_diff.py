# In this file, we will compute the expected profit difference between:
# i) Setting reserve as v0
# ii) Setting reserve as some r other than v0

import numpy as np
import AL_profits as AL
import non_equilibrium_profits as NEP
from pulp import LpMaximize, LpMinimize, LpProblem, LpVariable, PULP_CBC_CMD

def compute_expected_profit_diff(n,r,r0,v0,
                                 v_nn,G_hat_vnn,
                                 b_nns, b_n1ns,
                                ):
    fnnL_r, fnnU_r = AL.Fnn_KDE(n,r, v_nn, G_hat_vnn)
    fnnL_r0, fnnU_r0 = AL.Fnn_KDE(n,r0, v_nn, G_hat_vnn)
    
    i1 = np.mean([max(r,b) for b in b_nns])
    i2 = np.mean([max(r0,b) for b in b_n1ns])
    i3 = np.mean([max(r,b) for b in b_n1ns])
    i4 = np.mean([max(r0,b) for b in b_nns])
    
    ub = (i1-i2) - (r-v0)*fnnL_r + (r0-v0)*fnnU_r0
    lb = (i3-i4) - (r-v0)*fnnU_r + (r0-v0)*fnnL_r0
    return lb, ub

def LPsolver(r,r0,v0,
             fnnL_r, fnnU_r, fnnL_r0, fnnU_r0, 
             XgeqY=True, max=True,extraConstraint=False):
    # LP-Solver
    if max==True:
        prob = LpProblem("Profit Difference", LpMaximize)
    else:   # Minimize objecitve
        prob = LpProblem("Profit Difference", LpMinimize)
    X = LpVariable("X", lowBound=fnnL_r, upBound=fnnU_r)
    Y = LpVariable("Y", lowBound=fnnL_r0, upBound=fnnU_r0)
    # Obj. Function
    prob += -(r-v0)*X + (r0-v0)*Y
    # Constraints
    if extraConstraint:
        if XgeqY:
            prob += (Y <= X, "Y_leq_X_constraint")
        else:
            prob += (X <= Y, "X_leq_Y_constraint")
    
    status = prob.solve(PULP_CBC_CMD(msg=0))
    return prob.objective.value()

def compute_expected_profit_diff_r0(n,r,r0,v0,
                                    v_nn,G_hat_vnn,
                                    b_nns, b_n1ns,
                                    ):
    fnnL_r, fnnU_r = AL.Fnn_KDE(n,r, v_nn, G_hat_vnn)
    if fnnL_r > fnnU_r:
        fnnL_r = fnnU_r
    fnnL_r0, fnnU_r0 = AL.Fnn_KDE(n,r0, v_nn, G_hat_vnn)
    i1 = np.mean([max(r,b) for b in b_nns])
    i2 = np.mean([max(r0,b) for b in b_n1ns])
    i3 = np.mean([max(r,b) for b in b_n1ns])
    i4 = np.mean([max(r0,b) for b in b_nns])
    
    if r>=r0:
        B_u = LPsolver(r,r0,v0,fnnL_r, fnnU_r, fnnL_r0, fnnU_r0, XgeqY=True, max=True)
        B_l = LPsolver(r,r0,v0,fnnL_r, fnnU_r, fnnL_r0, fnnU_r0, XgeqY=True, max=False)
    else:
        B_u = LPsolver(r,r0,v0,fnnL_r, fnnU_r, fnnL_r0, fnnU_r0, XgeqY=False, max=True)
        B_l = LPsolver(r,r0,v0,fnnL_r, fnnU_r, fnnL_r0, fnnU_r0, XgeqY=False, max=False)
        
    ub = (i1-i2) + B_u
    lb = (i3-i4) + B_l
    return lb, ub

def exact_profit_diff(n,r,r0, max_bid_dicts, v_nn, g_hat_v, G_hat_vnn, v0):
    exp_bnn_geq_r = NEP.calc_positive_term(max_bid_dicts, n, r, v_nn, g_hat_v)
    exp_bnn_geq_r0 = NEP.calc_positive_term(max_bid_dicts, n, r0, v_nn, g_hat_v)
    term1 = exp_bnn_geq_r - exp_bnn_geq_r0
    
    term2 = v0 * (G_hat_vnn[NEP.find_nearest_below(v_nn, r0)]
                  - G_hat_vnn[NEP.find_nearest_below(v_nn, r)])
    
    return term1 - term2
