import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.optimize import minimize
import seaborn as sns
import matplotlib
matplotlib.rcParams.update({'font.size': 16})
from tqdm import tqdm

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


# phi function
def integrate_0_to_phi(n,phi):
    def f(s): 
        return (s**(n-2) * (1-s))
    return n * (n-1) * integrate.quad(f,0,phi)[0]

def calc_phi(n,H):
    bds = [(0,1)]
    return minimize(lambda phi: (integrate_0_to_phi(n,phi)-H)**2, x0=[0.7], method="Powell", bounds = bds).x[0]


def LB_Fnn(v, n, nBar, values_dict):
    ns = list(values_dict.keys())
    first_term = 0
    for m in [m for m in ns if m>n and m<=nBar]:
        first_term += n/m/(m-1) * calc_Fn1nhat(values_dict[m],v)
    second_term = n/nBar * calc_phi(nBar, calc_Fn1nhat(values_dict[nBar],v))**nBar
    return first_term + second_term


def plot(OUTPATH, n, n_range, rho, mu, sigma, num_sims=10000):
    values_dict = simulate_values_alln(n_range, rho, mu, sigma, num_sims)
    # all_values = []
    # for m in n_range:
    #     for vals in values_dict[m]:
    #         all_values.extend(vals)
    # v_range = np.linspace(min(all_values),max(all_values), 1000)
    v_range = np.linspace(0, 60, 1000)
    
    sns.set_style('darkgrid')
    plt.figure(figsize=(10,6), tight_layout=True)
    for nBar in tqdm(n_range[2:]):
        plt.plot(v_range, [LB_Fnn(v, n, nBar, values_dict) for v in v_range], label=r"$\bar{n}=v1$".replace('v1',str(nBar)))
    plt.legend()

    plt.title(r'Lower Bounds for $F_{3:3}$; $\rho$=v1; Lognormal(v2,v3)'.replace('v1',str(rho)).replace('v2',str(mu)).replace('v3',str(sigma)))
    plt.savefig(OUTPATH + "FnnLBs_corr={}.png".format(rho))
    return

OUT_PATH = "/Users/brucewen/Desktop/honors_thesis/estimation/simulations/varyingN_corr_whenIntsct/results/"
plot(OUT_PATH, n=3, n_range=[2,3,4,5,6,7,8,9,10], rho=0, mu=2.5, sigma=0.5, num_sims=10000)
plot(OUT_PATH, n=3, n_range=[2,3,4,5,6,7,8,9,10], rho=0.3, mu=2.5, sigma=0.5, num_sims=10000)
plot(OUT_PATH, n=3, n_range=[2,3,4,5,6,7,8,9,10], rho=0.5, mu=2.5, sigma=0.5, num_sims=10000)
plot(OUT_PATH, n=3, n_range=[2,3,4,5,6,7,8,9,10], rho=0.7, mu=2.5, sigma=0.5, num_sims=10000)
plot(OUT_PATH, n=3, n_range=[2,3,4,5,6,7,8,9,10], rho=1, mu=2.5, sigma=0.5, num_sims=10000)