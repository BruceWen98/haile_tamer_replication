import haile_tamer_simulation as HTS
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.special as sp
import collections
import AL_HT_bounds as AHb
import seaborn as sns
import pickle
import random


np.random.seed(2022)

def run_T_simulations_varyingN(T, lbda, increment, distribution):
    all_values = []
    all_bids = []
    max_bid_dicts = []
    for i in tqdm(range(T)):
        n = np.random.randint(2, 10)
        try: 
            values, bids, losing_player, max_bid_dict = HTS.run_1_simulation(n, lbda, increment, distribution)
        except Exception as e:
            # Case of less than 2 players
            continue
        all_values.append(values)
        all_bids.append(bids)
        
        max_bid_dict = dict(collections.OrderedDict(sorted(max_bid_dict.items())))  # Sort the dict by key, ascending order.
        max_bid_dicts.append(max_bid_dict)

    return all_values, all_bids, max_bid_dicts

all_values, all_bids, max_bid_dicts = run_T_simulations_varyingN(T=10000, lbda=0.1, increment=0.1, distribution='normal')


# Take only top 2 bids.
def top2(d):
    top2keys = sorted(d, key=d.get, reverse=True)[:2]
    out_dict = {}
    out_dict["1"] = d[top2keys[0]]
    out_dict["2"] = d[top2keys[1]]
    out_dict["n"] = len(d)
    return out_dict

data = [top2(max_bid_dict) for max_bid_dict in max_bid_dicts]
data = [d for d in data if d['2']>0.4]
print(data)

X = np.linspace(0.0000001, 10, 100)
n=2

# PATH = "/Users/brucewen/Desktop/honors_thesis/estimation/sotheby_data/with_N_bidders/"
# VAL = "withN"
OUTPATH = "/Users/brucewen/Desktop/honors_thesis/estimation/simulations/compare_phi/figs/"

# with open(PATH + 'output_dicts_ALL_{}.pickle'.format(VAL), 'rb') as handle:
#     data = pickle.load(handle)


def comp(data):
    list1 = []
    list2 = []
    gn1ns = []
    gnns = []
    for v in X:
        gn1n = AHb.Gn1n(n,v,data)
        gnn = AHb.Gnn(n,v,data)
        phin1nGn1n = AHb.calc_phi( n, AHb.Gn1n(n,v,data) )
        phinnGnn = AHb.calc_phiNN( n, AHb.Gnn(n,v,data) )
        list1.append(phin1nGn1n)
        list2.append(phinnGnn)
        gn1ns.append(gn1n)
        gnns.append(gnn)
    

    
    sns.set_style('darkgrid')
    plt.figure(figsize=(10,6), tight_layout=True)
    plt.xlabel("Value")
    plt.ylabel("CDF")
    
    ### Different Distributions
    ## Lognormal
    # lognormal_mu = 1
    # lognormal_sigma = 0.1
    # plt.title("Simulated Data: Distribution={}, parameters={},{}".format("lognormal", lognormal_mu, lognormal_sigma))
    
    ## Uniform
    # plt.title("Simulated Data: Distribution={}".format("Uniform"))
    
    ## Beta
    # alpha = 0.9
    # beta = 0.9
    # plt.title("Simulated Data: Distribution={}, parameters={},{}".format("Beta", alpha, beta))
    
    ## Beta Bimodal
    # alpha1, beta1, alpha2, beta2 = 1.1,5,20,2
    # plt.title("Simulated Data: Distribution={}, parameters={},{},{},{}".format("Bimodal Beta", alpha1, beta1, alpha2, beta2))

    ## Normal
    mu, sigma = 1,5

    plt.scatter(X, gn1ns, label=r"$G_{n-1:n}$")
    plt.scatter(X, gnns, label=r"$G_{n:n}$")
    plt.scatter(X,list1,color='orange', label=r"$\phi_{n-1:n}(G_{n-1:n}(v))$", marker='x')
    plt.scatter(X,list2,color='purple', label=r"$\phi_{n:n}(G_{n:n}(v))$", marker='^')
    
    # cdf = 1/2 * ( 1 + sp.erf((np.log(X) - lognormal_mu) / (np.sqrt(2) * lognormal_sigma)) )
    # plt.plot(X, cdf, linewidth=2, color='r', label="CDF of lognormal({},{})".format(lognormal_mu,lognormal_sigma))
    
    # cdf_bimodalBeta = sp.betainc(alpha1,beta1,X)/2 + sp.betainc(alpha2,beta2,X)/2
    # plt.plot(X, cdf_bimodalBeta, linewidth=2, color='r', label="CDF of Bimodal Beta({},{},{},{})".format(alpha1,beta1,alpha2,beta2))
    
    plt.legend()
    # plt.show()
    # plt.savefig(OUTPATH + "lognormal_{}_{}.png".format(lognormal_mu, lognormal_sigma))
    # plt.savefig(OUTPATH + "uniform.png")
    # plt.savefig(OUTPATH + "beta_{}_{}.png".format(alpha, beta))
    # plt.savefig(OUTPATH + "bimodalBeta_{}_{}_{}_{}.png".format(alpha1, beta1, alpha2, beta2))
    plt.savefig(OUTPATH + "normal_{}_{}.png".format(mu, sigma))
    
comp(data)