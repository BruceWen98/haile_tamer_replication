import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.optimize import minimize
import seaborn as sns

def generate_correlated_lognormal(n, mu, sigma, rho):
    '''
    n: number of values to generate
    mu: mean of the logarithm of the values
    sigma: standard deviation of the logarithm of the values
    rho: correlation coefficient
    '''
    # Generate covariance matrix
    cov_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                cov_matrix[i,j] = sigma**2
            else:
                cov_matrix[i,j] = sigma * sigma * rho
    
    # Generate multivariate lognormal distribution
    mean = np.ones(n) * mu
    correlated_lognormal = np.random.multivariate_normal(mean, cov_matrix)
    return np.exp(correlated_lognormal)

def simulate_values_alln(n_range, rho, mu, sigma, num_sims=10000):
    values_dict = {}
    for n in n_range:
        values_dict[n] = [generate_correlated_lognormal(n, mu, sigma, rho) for i in range(num_sims)]
    return values_dict

def calc_Fn1nhat(values_list,v):
    count = 0
    Tn = len(values_list)
    for values in values_list:
        # 2nd highest bid in the bid dictionary. Note that i are n-1, n.
        secondHighest_bid = sorted(values, reverse=True)[1]
        if secondHighest_bid <= v:
            count+=1
        continue
    return count/Tn

values_dict = simulate_values_alln([3,4,5], 1, 0, 1, 1000)

vs = np.linspace(0, 20, 1000)
for n in [3,4,5]:
    plt.plot(vs, [calc_Fn1nhat(values_dict[n], v) for v in vs], label=f'n={n}')
    
plt.show()