from scipy.stats import norm
from scipy.optimize import minimize
import numpy as np
import math
import scipy.integrate as integrate

##### 0. Numerical Integration of the pdf of the n-1:n order statistic. #####
def integrate_0_to_phi(n,phi):
    def f(s): 
        return (s**(n-2) * (1-s))
    return n * (n-1) * integrate.romberg(f,0,phi)

def calc_phi(n,H):
    bds = [(0,1)]
    return minimize(lambda phi: (integrate_0_to_phi(n,phi)-H)**2, x0=[0.7], method="Powell", bounds = bds).x[0]

def der_phi(n,phi):
    return 1/ ( n * (n-1) * (phi**(n-2)- phi**(n-1)) )

##### 1. AL Inference Functions #####
def sigma_hat_low2(max_bid_dicts, n, r, v0):
    max_bid_dicts_n = [max_bid_dict for max_bid_dict in max_bid_dicts if max_bid_dict['n'] == n]
    bn1ns = np.array([max_bid_dicts_n['2'] for max_bid_dicts_n in max_bid_dicts_n])
    bnns = np.array([max_bid_dicts_n['1'] for max_bid_dicts_n in max_bid_dicts_n])
    
    indicatorBnn = np.array([1 if bnn<=r else 0 for bnn in bnns])
    Sigma = np.cov([max(r,bn1n) for bn1n in bn1ns], indicatorBnn)
    J = np.array([1, -(r-v0)])
    
    return J@(Sigma)@(J.T)      # delta method

def sigma_hat_high2(max_bid_dicts, n, r, v0, phi):
    max_bid_dicts_n = [max_bid_dict for max_bid_dict in max_bid_dicts if max_bid_dict['n'] == n]
    bnns = np.array([max_bid_dicts_n['1'] for max_bid_dicts_n in max_bid_dicts_n])
    
    indicatorBnn = np.array([1 if bnn<=r else 0 for bnn in bnns])
    Sigma = np.cov([max(r,bnn) for bnn in bnns], indicatorBnn)
    # H = np.mean(bnns)
    # phi = calc_phi(n,H)
    der = n* phi**(n-1) * der_phi(n, phi)
    
    J = np.array([1, -(r-v0)*der])
    
    return J@(Sigma)@(J.T)      # delta method
    
def sigma_hat_low2_varyingN(max_bid_dicts, n, r, v0, phi=None, Fm1ms=None):
    unique_n_values = np.unique([max_bid_dict['n'] for max_bid_dict in max_bid_dicts])
    above_n_values = [m for m in unique_n_values if m >= n]
    bid_matrices = {m: np.array([max_bid_dict['2'] for max_bid_dict in max_bid_dicts if max_bid_dict['n'] == m]) for m in above_n_values}
    max_bid_dicts = [mbd for mbd in max_bid_dicts if mbd['n'] >= n]
    T = len(max_bid_dicts)
    
    mat = np.zeros((len(above_n_values) * 2, T))

    def prob(m):
        return len(bid_matrices[m]) / T

    def ind(bid, r):
        return 1 if bid <= r else 0

    def pop_first2rows(mat):
        row1 = [1 if mbd['n'] == n else 0 for mbd in max_bid_dicts]
        row2 = [max(r,mbd['2']) if mbd['n'] == n else 0 for mbd in max_bid_dicts]
        mat[0] = row1
        mat[1] = row2
        return mat
    
    mat = pop_first2rows(mat)

    def populate_rows(start_index, m_value):
        row_1 = [1 if mbd['n'] == m_value else 0 for mbd in max_bid_dicts]
        row_2 = [ind(mbd['2'], r) if mbd['n'] == m_value else 0 for mbd in max_bid_dicts]
        mat[start_index] = row_1
        mat[start_index + 1] = row_2

    for index, m in enumerate(above_n_values[1:], start=1):
        populate_rows(2 * index, m)

    Sigma = np.cov(mat)

    J = np.zeros(len(above_n_values) * 2)
    J[0] = - 1/prob(n)**2 * np.mean([max(r, mbd['2']) if mbd['n'] == n else 0 for mbd in max_bid_dicts])
    J[1] = 1/prob(n)

    for index, m in enumerate(above_n_values[1:-2]):
        J[2 + 2 * index] = (r - v0) * 1/prob(m)**2 * n / m / (m - 1) * np.mean([ind(mbd['2'], r) if mbd['n'] == m else 0 for mbd in max_bid_dicts])
        J[2 + 2 * index + 1] = -(r - v0) * n / m / (m - 1) * 1/prob(m)

    max_n = max(above_n_values)
    J[-2] = 0#(r - v0) * (n/(max_n-1)/max_n + n / max_n) * 1/prob(max_n)**2 * np.mean([ind(mbd['2'], r) if mbd['n'] == max_n else 0 for mbd in max_bid_dicts])
    J[-1] = 0#-(r - v0) * (n/(max_n-1)/max_n + n / max_n) * 1/prob(max_n)

    out = J @ Sigma @ J.T
    if out < 0:
        return 0
    else:
        return out

def sigma_hat_high2_varyingN(max_bid_dicts, n, r, v0, phi=None, Fm1ms=None):
    unique_n_values = np.unique([max_bid_dict['n'] for max_bid_dict in max_bid_dicts])
    above_n_values = [m for m in unique_n_values if m >= n]
    bid_matrices = {m: np.array([max_bid_dict['1'] for max_bid_dict in max_bid_dicts if max_bid_dict['n'] == m]) for m in above_n_values}
    max_bid_dicts = [mbd for mbd in max_bid_dicts if mbd['n'] >= n]
    T = len(max_bid_dicts)

    mat = np.zeros((len(above_n_values) * 2, T))

    def prob(m):
        return len(bid_matrices[m]) / T

    def ind(bid, r):
        return 1 if bid <= r else 0

    def pop_first2rows(mat):
        row1 = [1 if mbd['n'] == n else 0 for mbd in max_bid_dicts]
        row2 = [max(r,mbd['1']) if mbd['n'] == n else 0 for mbd in max_bid_dicts]
        mat[0] = row1
        mat[1] = row2
        return mat
    
    mat = pop_first2rows(mat)

    def populate_rows(start_index, m_value):
        row_1 = [1 if mbd['n'] == m_value else 0 for mbd in max_bid_dicts]
        row_2 = [ind(mbd['1'], r) if mbd['n'] == m_value else 0 for mbd in max_bid_dicts]
        mat[start_index] = row_1
        mat[start_index + 1] = row_2

    for index, m in enumerate(above_n_values[1:], start=1):
        populate_rows(2 * index, m)

    Sigma = np.cov(mat)

    J = np.zeros(len(above_n_values) * 2)
    J[0] = - 1/prob(n)**2 * np.mean([max(r, mbd['1']) if mbd['n'] == n else 0 for mbd in max_bid_dicts])
    J[1] = 1/prob(n)

    for index, m in enumerate(above_n_values[1:-2]):
        J[2 + 2 * index] = (r - v0) * 1/prob(m)**2 * n / m / (m - 1) * np.mean([ind(mbd['2'], r) if mbd['n'] == m else 0 for mbd in max_bid_dicts])
        J[2 + 2 * index + 1] = -(r - v0) * n / m / (m - 1) * 1/prob(m)

    max_n = max(above_n_values)
    # H = np.mean(bid_matrices[max(above_n_values)])
    # phi = calc_phi(max(above_n_values),H)
    der = n* phi**(max(above_n_values)-1) * der_phi(max(above_n_values), phi)    
    J[-2] = 0#(r - v0) * 1/prob(max_n)**2 * (n/(max_n-1)/max_n * np.mean([ind(mbd['1'], r) if mbd['n'] == m else 0 for mbd in max_bid_dicts]) + n / max_n * phi**(max(above_n_values)))
    J[-1] = 0#-(r - v0) * 1/prob(max_n) * (n/(max_n-1)/max_n + der)
    out = J @ Sigma @ J.T
    if out < 0:
        return 0
    else:
        return out

def lambda_hat(pi_U, pi_L):
    return pi_U - pi_L

def solve_c_alpha(L, Lambda_hat, sigma_hat_Up, sigma_hat_Low, alpha=0.05):
    def f(c_alpha):
        A = np.sqrt(L) * Lambda_hat / max(sigma_hat_Up, sigma_hat_Low)
        return norm.cdf(c_alpha + A) - norm.cdf(-c_alpha) - 1 + alpha
    return minimize(lambda x: f(x)**2, x0=[0.05], method='Nelder-Mead').x[0]

def CI_lb(pi_L, c_alpha, sigma_hat_Low, L):
    return pi_L - c_alpha * sigma_hat_Low / np.sqrt(L)

def CI_ub(pi_U, c_alpha, sigma_hat_Up, L):
    return pi_U + c_alpha * sigma_hat_Up / np.sqrt(L)


def CI_out(pi_U, pi_L, L, max_bid_dicts, n, r, v0, phi, alpha=0.05):
    sigma_hat_Up = math.sqrt(sigma_hat_high2(max_bid_dicts, n, r, v0, phi))
    sigma_hat_Low = math.sqrt(sigma_hat_low2(max_bid_dicts, n, r, v0))
    Lambda_hat = lambda_hat(pi_U, pi_L)
    c_alpha = solve_c_alpha(L, Lambda_hat, sigma_hat_Up, sigma_hat_Low, alpha)
    ci_lb = CI_lb(pi_L, c_alpha, sigma_hat_Low, L)
    ci_ub = CI_ub(pi_U, c_alpha, sigma_hat_Up, L)
    return ci_lb, ci_ub


def CI_out_varyingN(pi_U, pi_L, L, max_bid_dicts, n, r, v0, phi,Fm1ms, alpha=0.05):
    sigma_hat_Up = math.sqrt(sigma_hat_high2_varyingN(max_bid_dicts, n, r, v0, phi, Fm1ms))
    sigma_hat_Low = math.sqrt(sigma_hat_low2_varyingN(max_bid_dicts, n, r, v0, phi, Fm1ms))
    Lambda_hat = lambda_hat(pi_U, pi_L)
    c_alpha = solve_c_alpha(L, Lambda_hat, sigma_hat_Up, sigma_hat_Low, alpha)
    ci_lb = CI_lb(pi_L, c_alpha, sigma_hat_Low, L)
    ci_ub = CI_ub(pi_U, c_alpha, sigma_hat_Up, L)
    return ci_lb, ci_ub



# MIUK2 = 1/2/np.sqrt(np.pi)    

# def CI_lb_h(pi_L, c_alpha, sigma_hat_Low, L,h):
#     return pi_L - c_alpha * sigma_hat_Low / np.sqrt(L*h)

# def CI_ub_h(pi_U, c_alpha, sigma_hat_Up, L,h):
#     return pi_U + c_alpha * sigma_hat_Up / np.sqrt(L*h)

# def CI_out_varyingN(pi_U, pi_L, L, max_bid_dicts, n, r, v0, phi,Fm1ms, h, alpha=0.05):
#     sigma_hat_Up = math.sqrt(pi_U * MIUK2)
#     sigma_hat_Low = math.sqrt(pi_L * MIUK2)
#     Lambda_hat = lambda_hat(pi_U, pi_L)
#     c_alpha = solve_c_alpha(L, Lambda_hat, sigma_hat_Up, sigma_hat_Low, alpha)
#     ci_lb = CI_lb_h(pi_L, c_alpha, sigma_hat_Low, L, h)
#     ci_ub = CI_ub_h(pi_U, c_alpha, sigma_hat_Up, L, h)
#     return ci_lb, ci_ub