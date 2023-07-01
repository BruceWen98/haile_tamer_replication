import numpy as np
import matplotlib.pyplot as plt
import pickle

max_bid_dicts = pickle.load(open('/Users/brucewen/Desktop/honors_thesis/estimation/combined_data/categorized_data_withXi/ALL_d_art_1712obs.p','rb'))
max_bid_dicts = max_bid_dicts['data']

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
        for j in range(T):
            mat[start_index][j] = row_1[j]
            mat[start_index + 1][j] = row_2[j]

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
    J[-2] = (r - v0) * (n/(max_n-1)/max_n + n / max_n) * 1/prob(max_n)**2 * np.mean([ind(mbd['2'], r) if mbd['n'] == max_n else 0 for mbd in max_bid_dicts])
    J[-1] = -(r - v0) * (n/(max_n-1)/max_n + n / max_n) * 1/prob(max_n)

    return J @ Sigma @ J.T

def sigma_hat_high2_varyingN_OLD(max_bid_dicts, n, r, v0, phi):
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
        for j in range(T):
            mat[start_index][j] = row_1[j]
            mat[start_index + 1][j] = row_2[j]

    for index, m in enumerate(above_n_values[1:], start=1):
        populate_rows(2 * index, m)

    Sigma = np.cov(mat)

    J = np.zeros(len(above_n_values) * 2)
    J[0] = - 1/prob(n)**2 * np.mean([max(r, mbd['1']) if mbd['n'] == n else 0 for mbd in max_bid_dicts])
    J[1] = 1/prob(n)

    for index, m in enumerate(above_n_values[1:-2]):
        J[2 + 2 * index] = (r - v0) * 1/prob(m)**2 * n / m / (m - 1) * np.mean([max(r, mbd['1']) if mbd['n'] == m else 0 for mbd in max_bid_dicts])
        J[2 + 2 * index + 1] = -(r - v0) * n / m / (m - 1) * 1/prob(m)

    max_n = max(above_n_values)
    # H = np.mean(bid_matrices[max(above_n_values)])
    # phi = calc_phi(max(above_n_values),H)
    der = n* phi**(max(above_n_values)-1) * der_phi(max(above_n_values), phi)    
    J[-2] = (r - v0) * (n/(max_n-1)/max_n + n / max_n) * 1/prob(max_n)**2 * phi**(max(above_n_values))
    J[-1] = -(r - v0) * 1/prob(max_n) * (n/(max_n-1)/max_n + der)

    return J @ Sigma @ J.T



print(sigma_hat_low2_varyingN(max_bid_dicts, 4, 10, 0.667))



def sigma_hat_low2(max_bid_dicts, n, r, v0):
    max_bid_dicts_n = [max_bid_dict for max_bid_dict in max_bid_dicts if max_bid_dict['n'] == n]
    bn1ns = np.array([max_bid_dicts_n['2'] for max_bid_dicts_n in max_bid_dicts_n])
    bnns = np.array([max_bid_dicts_n['1'] for max_bid_dicts_n in max_bid_dicts_n])
    
    indicatorBnn = np.array([1 if bnn<=r else 0 for bnn in bnns])
    Sigma = np.cov([max(r,bn1n) for bn1n in bn1ns], indicatorBnn)
    J = np.array([1, -(r-v0)])
    
    return J@(Sigma)@(J.T)  

print(sigma_hat_low2(max_bid_dicts, 4, 5, 0.667))

