import AL_HT_bounds as AHb
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import seaborn as sns

PATH = "/Users/brucewen/Desktop/honors_thesis/estimation/sotheby_data/with_N_bidders/"
VAL = "withN"
OUTPATH = "/Users/brucewen/Desktop/honors_thesis/estimation/AL_estimation/figs/HT_AL_bds/"

with open(PATH + 'output_dicts_ALL_{}.pickle'.format(VAL), 'rb') as handle:
    output_dicts_ALL = pickle.load(handle)
    

X = np.linspace(0, 15, 100)

## AL bounds
#Fn-1:n bounds, Aradillas-Lopez
def estimate_Fn1n_AL(X, data, n):
    F_L_array = []
    F_U_array = []
    for v in tqdm(X):
        F_L_v, F_U_v = AHb.Fn1n(n, v, data)
        F_L_array.append(F_L_v)
        F_U_array.append(F_U_v)
    return F_L_array, F_U_array

#Fn:n bounds, Aradillas-Lopez
def estimate_Fnn_AL(X, data, n):
    F_L_array = []
    F_U_array = []
    for v in tqdm(X):
        F_L_v, F_U_v = AHb.Fnn(n, v, data)
        F_L_array.append(F_L_v)
        F_U_array.append(F_U_v)
    return F_L_array, F_U_array

## HT bounds
#Fn-1:n bounds, Haile-Tamer
def estimate_Fn1n_HT(X, data, n):
    F_L_array = []
    F_U_array = []
    for v in tqdm(X):
        F_L_v, F_U_v = AHb.F_HT_n1n(n, data, v, 0)
        F_L_array.append(F_L_v)
        F_U_array.append(F_U_v)
    return F_L_array, F_U_array

def estimate_Fnn_HT(X, data, n):
    F_L_array = []
    F_U_array = []
    for v in tqdm(X):
        F_L_v, F_U_v = AHb.F_HT_nn(n, data, v, 0)
        F_L_array.append(F_L_v)
        F_U_array.append(F_U_v)
    return F_L_array, F_U_array



for N in list(np.unique([d['n'] for d in output_dicts_ALL])):
    print("N = {}".format(N))
    Fn1nAL_lb, Fn1nAL_ub = estimate_Fn1n_AL(X, output_dicts_ALL,N)
    FnnAL_lb, FnnAL_ub = estimate_Fnn_AL(X, output_dicts_ALL,N)
    Fn1nHT_lb, Fn1nHT_ub = estimate_Fn1n_HT(X, output_dicts_ALL,N)
    FnnHT_lb, FnnHT_ub = estimate_Fnn_HT(X, output_dicts_ALL,N)


    sns.set_style('darkgrid')
    plt.figure(figsize=(10,6), tight_layout=True)
    plt.xlabel("Value")
    plt.ylabel("CDF")
    plt.title("Bounds on Bidder Values, AL&HT ({})".format(VAL))

    plt.scatter(X,Fn1nAL_lb,color='orange', label='AL lb, Fn-1:n', marker='x')
    plt.scatter(X,Fn1nAL_ub,color='blue', label='AL ub, Fn-1:n', marker='x')
    plt.scatter(X,FnnAL_lb,color='red', label='AL lb, Fn:n', marker='x')
    plt.scatter(X,FnnAL_ub,color='green', label='AL ub, Fn:n', marker='x')
    plt.scatter(X,Fn1nHT_lb,color='purple', label='HT lb, Fn-1:n', marker='^')
    plt.scatter(X,Fn1nHT_ub,color='yellow', label='HT ub, Fn-1:n', marker='^')
    plt.scatter(X,FnnHT_lb,color='black', label='HT lb, Fn:n', marker='^')
    plt.scatter(X,FnnHT_ub,color='pink', label='HT ub, Fn:n', marker='^')
    plt.legend()

    # plt.show()
    plt.savefig(OUTPATH + "AL_HT_bounds_{}_N{}.png".format(VAL, N))