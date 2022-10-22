import numpy as np
import pickle
import matplotlib.pyplot as plt
import haile_tamer_estimation_withN as HTE
from tqdm import tqdm
import sys

VAL = "withN"


with open('output_dicts_jump2_{}.pickle'.format(VAL), 'rb') as handle:
    output_dicts_jump2 = pickle.load(handle)

with open('output_dicts_NOjump2_{}.pickle'.format(VAL), 'rb') as handle:
    output_dicts_NOjump2 = pickle.load(handle)

with open('output_dicts_ALL_{}.pickle'.format(VAL), 'rb') as handle:
    output_dicts_ALL = pickle.load(handle)

X = np.linspace(0, 10, 100)

def estimate_F(X, data):
    F_U_array = []
    F_L_array = []
    for v in tqdm(X):
        F_U_v = HTE.F_hat_U(data,v,1)
        F_U_array.append(F_U_v)
        F_L_v = HTE.F_hat_L(data,v,-1,increment=0.1)
        F_L_array.append(F_L_v)
    return F_U_array, F_L_array

all_F_U, all_F_L = estimate_F(X, output_dicts_ALL)
all_F_U_jump2, all_F_L_jump2 = estimate_F(X, output_dicts_jump2)
all_F_U_NOjump2, all_F_L_NOjump2 = estimate_F(X, output_dicts_NOjump2)

plt.scatter(X,all_F_U,color='orange', label='upper bound')
plt.scatter(X,all_F_L,color='blue', label='lower bound')
plt.scatter(X,all_F_U_jump2,color='red', label='jump2_upper')
plt.scatter(X,all_F_L_jump2,color='green', label='jump2_lower')
plt.scatter(X,all_F_U_NOjump2,color='purple', label='nojump_upper')
plt.scatter(X,all_F_L_NOjump2,color='yellow', label='nojump_lower')
plt.legend()
plt.xlabel("Value")
plt.ylabel("CDF")
plt.title("Bounds on Bidder Values ({})".format(VAL))
plt.show()
