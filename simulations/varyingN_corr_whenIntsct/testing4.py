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

def simulate_values(n, rho, mu, sigma, num_sims=1000):
    return [generate_correlated_lognormal(n, mu, sigma, rho) for i in range(num_sims)]

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

def calc_Fnnhat(values_list,v):
    count = 0
    Tn = len(values_list)
    for values in values_list:
        Highest_bid = sorted(values, reverse=True)[0]
        if Highest_bid <= v:
            count+=1
        continue
    return count/Tn

# phi function
def integrate_0_to_phi(n,phi):
    def f(s): 
        return (s**(n-2) * (1-s))
    return n * (n-1) * integrate.quad(f,0,phi)[0]

def calc_phi(n,H):
    bds = [(0,1)]
    return minimize(lambda phi: (integrate_0_to_phi(n,phi)-H)**2, x0=[0.7], method="Powell", bounds = bds).x[0]




vs = np.linspace(0, 50, 1000)
valuesn5 = simulate_values(5, 0.1, 2.5, 0.5)
valuesn4 = simulate_values(4, 0.1, 2.5, 0.5)

#1
# fn1n = np.array([calc_Fn1nhat(valuesn5, v) for v in vs])
# secondTerm = np.array([(5) * calc_Fnnhat(valuesn4, v) for v in vs])
# lastTerm = np.array([(5-1) * calc_phi(5, fn1n[i])**5 for i in range(len(fn1n))])

#2
# fn1n5 = np.array([calc_Fn1nhat(valuesn5, v) for v in vs])
# fn1n4 = np.array([calc_Fn1nhat(valuesn4, v) for v in vs])
# secondTerm = np.array([(5-1) * calc_phi(5, fn1n5[i])**5 for i in range(len(fn1n5))])
# lastTerm = np.array([(5) * calc_phi(4, fn1n4[i])**4 for i in range(len(fn1n4))])
# plt.plot(vs, secondTerm-lastTerm)
# plt.show()

#3 Fn1n >= phi(n, Fn1n)^(n)
fn1n5 = np.array([calc_Fn1nhat(valuesn5, v) for v in vs])
secondTerm = np.array([calc_phi(5, fn1n5[i])**5 for i in range(len(fn1n5))])
plt.plot(vs, fn1n5-secondTerm)
plt.show()

#4 phi(n-1, Fn2n1)^(n-1) >= phi(n, Fn1n)^(n)
# fn1n4 = np.array([calc_Fn1nhat(valuesn4, v) for v in vs])
# fn1n5 = np.array([calc_Fn1nhat(valuesn5, v) for v in vs])
# firstTerm = np.array([calc_phi(4, fn1n4[i])**4 for i in range(len(fn1n4))])
# secondTerm = np.array([calc_phi(5, fn1n5[i])**5 for i in range(len(fn1n4))])
# plt.plot(vs, firstTerm-secondTerm)
# plt.show()