import numpy as np
import pickle
import matplotlib.pyplot as plt
import haile_tamer_estimation as HTE
from tqdm import tqdm

with open('top2_bids.pickle', 'rb') as handle:
    top2_bids = pickle.load(handle)

with open('output_dicts_jump2.pickle', 'rb') as handle:
    output_dicts_jump2 = pickle.load(handle)

with open('output_dicts_NOjump2.pickle', 'rb') as handle:
    output_dicts_NOjump2 = pickle.load(handle)




X = np.linspace(0, 7, 100)

def estimate_F(X, data):
    F_U_array = []
    F_L_array = []
    for v in tqdm(X):
        F_U_v = HTE.F_hat_U(data,HTE.calc_M(data),v,0.1)
        F_U_array.append(F_U_v)
        F_L_v = HTE.F_hat_L(data,HTE.calc_M(data),v,-0.1,increment=0.1)
        F_L_array.append(F_L_v)
    return F_U_array, F_L_array

all_F_U, all_F_L = estimate_F(X, top2_bids)
all_F_U_jump2, all_F_L_jump2 = estimate_F(X, output_dicts_jump2)
all_F_U_NOjump2, all_F_L_NOjump2 = estimate_F(X, output_dicts_NOjump2)

plt.scatter(X,all_F_U,color='orange')
plt.scatter(X,all_F_L,color='blue')
plt.scatter(X,all_F_U_jump2,color='red')
plt.scatter(X,all_F_L_jump2,color='green')
plt.scatter(X,all_F_U_NOjump2,color='purple')
plt.scatter(X,all_F_L_NOjump2,color='yellow')
plt.show()
