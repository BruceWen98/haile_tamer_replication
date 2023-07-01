import pandas as pd
import numpy as np
import Est_profits as EST
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import sys
import os
from pathlib import Path

#TODO: UPDATE FONTSIZE OF PLOTS

### 1. Helping Functions
def compute_v0(dicts):
    ratios = []
    for d in dicts:
        ratios.append(d['low_rel_high'])
    return np.median(ratios)

def compute_bounds(X,n, data_dicts, UB_V, V0):
    profits_lb_AL = []
    profits_ub_AL = []
    for r in tqdm(X):
        p_AL_lb, p_AL_ub = EST.compute_expected_profit_KDE(n, r, data_dicts, UB_V, V0, variedN=False)
        profits_lb_AL.append(p_AL_lb)
        profits_ub_AL.append(p_AL_ub)
    return profits_lb_AL, profits_ub_AL
    
def getNamesForGraphs2(characteristics):
    # Category
    if characteristics['cat0']==None and characteristics['cat1']==None:
        cat = 'all'
    elif characteristics['cat0']!=None and characteristics['cat1']==None:
        cat = characteristics['cat0']
    else:
        cat = characteristics['cat0'] + ':' + characteristics['cat1']
    
    # Location
    if characteristics['loc'] == None:
        loc = 'all'
    else:
        loc = characteristics['loc'] 
    
    # Value
    if characteristics['value'] == None:
        val = 'all'
    else:
        val = characteristics['value']
    
    # Degree of Competition
    if characteristics['degree_competition'] == None:
        deg_competition = 'all'
    else:
        deg_competition = characteristics['degree_competition']
        
    return cat, loc, val, deg_competition

def run_AL_profits(INPATH, OUTPATH, RESERVE_PATH, UB_V=10, num_points=500):
    ### 2. Input Data
    with open(INPATH, 'rb') as handle:
        data_file = pickle.load(handle)
        data_dicts = data_file['data']
        characteristics = data_file['characteristics']
        
    cat, loc, val, deg_competition = getNamesForGraphs2(characteristics)
            
    print("Values of N = ", EST.calc_N_set(data_dicts))
    print("Size of this Data = ", len(data_dicts))

    V0 = compute_v0(data_dicts)
    X = np.linspace(V0, UB_V, num_points)

    ### 3. Profit against reserve price curve for each n 
    allbounds    = []
    allbounds_vN = []
    for n in list(np.unique([d['n'] for d in data_dicts])):
        print("n = {}".format(n))
        print(len([d for d in data_dicts if d['n'] == n]))
        try:
            profits_lb_AL, profits_ub_AL = compute_bounds(X,n, data_dicts, UB_V, V0)
        except ValueError as e:     #ValueError: Root finding did not converge. Need more data.
            print(e)
            continue

        obs = len([d for d in data_dicts if d['n'] >= n])
        sns.set_style('darkgrid')
        plt.figure(figsize=(10,6), tight_layout=True)
        plt.title(r"Category: {}, Loc: {}, Sell Price: {}, Competition: {};  N$\geq${}; Obs={}".format(cat,loc,val,deg_competition,n,obs))
        plt.xlabel("Reserve Price, rel. to High Estimate")
        plt.ylabel("Expected Profit, rel. to High Estimate")
        plt.scatter(X,profits_lb_AL,color='plum', label='lb', marker=6)    # marker=6 is a caretup
        plt.scatter(X,profits_ub_AL,color='darkorchid', label='ub', marker=7)      # marker=7 is a caretdown

        plt.legend()
        plt.savefig(OUTPATH + "profits_N_geq{}.png".format(n))

        allbounds.append({'N': n, 'lb': profits_lb_AL, 'ub': profits_ub_AL, 'count': len([d for d in data_dicts if d['n'] == n])})

    return


files_path = sys.argv[1]
p = Path(files_path)
files = [x for x in p.iterdir() if x.is_file()]

# # Stuff to Run
# torun = {
#     # 'ALL_d': 10,
#     # 'ALL_d_art': 10,
#     # 'ALL_d_art_imp20': 10,
#     # 'ALL_d_art_oldMasters': 5,
#     # 'ALL_d_hk_art_imp20': 10,
#     # 'ALL_d_nyc_art_chinese': 15,
#     # 'ALL_d_nyc_art_imp20': 10,
#     # 'ALL_d_par_art_imp20': 5,
#     # 'ALL_high_high': 5,
#     # 'ALL_high': 5,
#     # 'ALL_low_high': 15,
#     # 'ALL_low_low': 5,
#     # 'ALL_low_mid': 5,
#     # 'art_high': 5,
#     # 'art_imp20_mid': 10,
#     # 'art_imp20_nyc_low': 10,
#     # 'art_imp20_nyc_mid': 5,
#     'art_imp20_nyc_mid_mid': 3,         # very nice!
#     # 'art_low': 15,
# }
# files = [f for f in files if f.name.rsplit("_",1)[0] in torun.keys()]

for f in tqdm(files):
    if str(f)[-2:]==".p":
        print("Working on {} now...".format(f))
        INPATH = str(f)
        OUTPATH = "/Users/brucewen/Desktop/honors_thesis/estimation/HigherN_estimation/categorized_data_results_withXi/" + INPATH.split("/")[-1].split(".")[0] + "/"
        if Path(OUTPATH).is_dir()==False:
            os.mkdir(OUTPATH)
        
        # ub_v = torun[f.name.rsplit("_",1)[0]]
        ub_v = 10
        print("UB_V: {}".format(ub_v))
        run_AL_profits(INPATH, OUTPATH, "", UB_V=ub_v, num_points=200)
    else:
        continue