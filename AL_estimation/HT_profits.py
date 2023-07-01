import AL_HT_bounds as AHb
import numpy as np

##### Calculate Expected Revenue using Law of Large Numbers (LLN). #####
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

##### Estimation of the Expected Profit (using HT bounds) #####
# using LLN method (expected value)
def compute_expected_profit_LLN(n, r, max_bid_dicts, ub_v, v0):
    f = lambda v: AHb.F_HT_n1n(n, max_bid_dicts, v)
    vf = np.vectorize(f)
    V_support = np.linspace(0, ub_v, 1000)

    Gnn,phi_inv_Gn1n = vf(V_support)
    
    exp_rev_ub = computeExp(r, 100000, lambda x: V_support[closestXidx(x,Gnn)])     # Number of draws for LLN = 100,000
    exp_rev_lb = computeExp(r, 100000, lambda x: V_support[closestXidx(x,phi_inv_Gn1n)])
    
    fnn_l, fnn_u = AHb.F_HT_nn(n, max_bid_dicts, r)

    lb = exp_rev_lb - v0 - fnn_u * (r - v0)
    ub = exp_rev_ub - v0 - fnn_l * (r - v0)
    print("Expected Profit bounds: ", lb, ub)
    return lb,ub