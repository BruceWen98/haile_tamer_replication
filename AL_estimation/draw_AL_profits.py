import pandas as pd
import numpy as np
import AL_profits as AL
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import AL_unconditionalN as ALUN
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

def get_KDE_parameters(data_dicts, n):
    v_n1n, ghat_KDE_n1n, h = AL.ghat_KDE(data_dicts,n-1,n, ker='gaussian', bandwidth="ISJ")
    v_nn, ghat_KDE_nn, h = AL.ghat_KDE(data_dicts,n,n, ker='gaussian', bandwidth="ISJ")
    G_hat_vnn = AL.KDE_pdf_to_cdf(v_nn, ghat_KDE_nn)
    return v_n1n, ghat_KDE_n1n, v_nn, ghat_KDE_nn, G_hat_vnn

def compute_bounds(X,n, data_dicts, ub_v, v0):
    profits_lb_AL = []
    profits_ub_AL = []
    
    # Get the KDE parameters
    v_n1n, ghat_KDE_n1n, v_nn, ghat_KDE_nn, G_hat_vnn = get_KDE_parameters(data_dicts, n)
    
    for r in tqdm(X):
        p_AL_lb, p_AL_ub = AL.compute_expected_profit_KDE(n, r, data_dicts, ub_v, v0, 
                                                          v_n1n,ghat_KDE_n1n,v_nn,ghat_KDE_nn,G_hat_vnn,
                                                          integral_method=False, variedN=False, KDEs=None)
        profits_lb_AL.append(p_AL_lb)
        profits_ub_AL.append(p_AL_ub)
    return profits_lb_AL, profits_ub_AL

def compute_bounds_vN(X,n, data_dicts, ub_v, v0, KDEs):
    profits_lb_AL_vN = []
    profits_ub_AL_vN = []
    
    # Get the KDE parameters
    v_n1n, ghat_KDE_n1n, v_nn, ghat_KDE_nn, G_hat_vnn = get_KDE_parameters(data_dicts, n)
    
    for r in tqdm(X):
        p_AL_lb, p_AL_ub = AL.compute_expected_profit_KDE(n, r, data_dicts, ub_v, v0, 
                                                          v_n1n,ghat_KDE_n1n,v_nn,ghat_KDE_nn,G_hat_vnn,
                                                          integral_method=False, variedN=True, KDEs=KDEs)
        profits_lb_AL_vN.append(p_AL_lb)
        profits_ub_AL_vN.append(p_AL_ub)
    return profits_lb_AL_vN, profits_ub_AL_vN

def getNamesForGraphs(INPATH):
    # Name for Graphs
    name_data = INPATH.split('/')[-1][6:-2]
    print(name_data)
    if len(name_data) >= 5:
        if name_data[-4:] == '_low' or name_data[-4:] == '_mid':
            cat = name_data[:-4]
            val = name_data[-3:]
        elif name_data[-5:] == '_high':
            cat = name_data[:-5]
            val = name_data[-4:]
        else:
            cat = name_data
            val = 'all'
    else:
        cat = name_data
        val = 'all'
    
    return cat, val
    
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

def run_AL_profits(INPATH, OUTPATH, RESERVE_PATH, UB_V=15, num_points=1000):
    ### 2. Input Data
    with open(INPATH, 'rb') as handle:
        data_file = pickle.load(handle)
        data_dicts = data_file['data']
        characteristics = data_file['characteristics']
        
    cat, loc, val, deg_competition = getNamesForGraphs2(characteristics)
            
    print("Values of N = ", AL.calc_N_set(data_dicts))
    print("Size of this Data = ", len(data_dicts))

    V0 = compute_v0(data_dicts)
    X = np.linspace(V0, UB_V, num_points)

    KDEs = AL.getAll_KDEs(2,data_dicts)     # comment if not using variedN


    ### 3. Profit against reserve price curve for each n 
    allbounds    = []
    allbounds_vN = []
    for n in list(np.unique([d['n'] for d in data_dicts])):
        print("n = {}".format(n))
        print(len([d for d in data_dicts if d['n'] == n]))
        profits_lb_AL_vN, profits_ub_AL_vN = None, None
        if n == max([d['n'] for d in data_dicts]):
            try:
                profits_lb_AL, profits_ub_AL = compute_bounds(X,n, data_dicts, UB_V, V0)
            except ValueError as e:     #ValueError: Root finding did not converge. Need more data.
                print(e)
                continue
        else:
            try: 
                profits_lb_AL, profits_ub_AL = compute_bounds(X,n, data_dicts, UB_V, V0)
                profits_lb_AL_vN, profits_ub_AL_vN = compute_bounds_vN(X,n, data_dicts, UB_V, V0, KDEs)
            except ValueError as e:     #ValueError: Root finding did not converge. Need more data.
                print(e)
                continue

        obs = len([d for d in data_dicts if d['n'] == n])
        sns.set_style('darkgrid')
        plt.figure(figsize=(10,6), tight_layout=True)
        plt.title("Category: {}, Loc: {}, Sell Price: {}, Competition: {};  N={}; Obs={}".format(cat,loc,val,deg_competition,n,obs))
        plt.xlabel("Reserve Price, rel. to High Estimate")
        plt.ylabel("Expected Profit, rel. to High Estimate")
        plt.scatter(X,profits_lb_AL,color='plum', label='lb', marker=6)    # marker=6 is a caretup
        plt.scatter(X,profits_ub_AL,color='darkorchid', label='ub', marker=7)      # marker=7 is a caretdown
        if profits_lb_AL_vN is not None:
            plt.scatter(X,profits_lb_AL_vN,color='limegreen', label='lb varying N', marker=6)    # marker=6 is a caretup
            plt.scatter(X,profits_ub_AL_vN,color='seagreen', label='ub varying N', marker=7)      # marker=7 is a caretdown
            allbounds_vN.append({'N': n, 'lb': profits_lb_AL_vN, 'ub': profits_ub_AL_vN, 'count': len([d for d in data_dicts if d['n'] == n])})

        plt.legend()
        # plt.show()
        plt.savefig(OUTPATH + "profits_n{}.png".format(n))

        allbounds.append({'N': n, 'lb': profits_lb_AL, 'ub': profits_ub_AL, 'count': len([d for d in data_dicts if d['n'] == n])})
        

    # ### 4. Unconditional N bounds
    # unconditionalN_lb, unconditionalN_ub = ALUN.get_unconditionalN_bounds(allbounds, len(X))
    # unconditionalN_lb_vN, unconditionalN_ub_vN = ALUN.get_unconditionalN_bounds(allbounds_vN, len(X))

    # sns.set_style('darkgrid')
    # plt.figure(figsize=(10,6), tight_layout=True)
    # plt.title("Auction Category: {}; Transaction Price Range: {}, Unconditional N".format(cat,val))
    # plt.xlabel("Reserve Price")
    # plt.ylabel("Expected Profit")
    # plt.scatter(X,unconditionalN_lb,color='plum', label='lb', marker=6)    # marker=6 is a caretup
    # plt.scatter(X,unconditionalN_ub,color='darkorchid', label='ub', marker=7)      # marker=7 is a caretdown
    # plt.scatter(X,unconditionalN_lb_vN,color='limegreen', label='lb varying N', marker=6)    # marker=6 is a caretup
    # plt.scatter(X,unconditionalN_ub_vN,color='seagreen', label='ub varying N', marker=7)      # marker=7 is a caretdown
    # plt.legend()
    # plt.savefig(OUTPATH + "profits_unconditionalN.png".format(n))

    # # Save the unconditional N data for a single Optimal Reserve Price
    # bounds_unconditionalN = {
    #     'X': X,
    #     'lb': unconditionalN_lb,
    #     'ub': unconditionalN_ub
    # }
    # bounds_unconditionalN_vN = {
    #     'X': X,
    #     'lb': unconditionalN_lb_vN,
    #     'ub': unconditionalN_ub_vN
    # }
    # pickle.dump(bounds_unconditionalN, open(RESERVE_PATH + OUTPATH.split("/")[-2] + "_bounds_unconditionalN.p".format(n), "wb"))
    # pickle.dump(bounds_unconditionalN_vN, open(RESERVE_PATH + OUTPATH.split("/")[-2] + "_bounds_unconditionalN_vN.p".format(n), "wb"))

    return


files_path = sys.argv[1]
p = Path(files_path)
files = [x for x in p.iterdir() if x.is_file()]
SELECT_RESERVE_PATH = "/Users/brucewen/Desktop/honors_thesis/selecting reserve/code/computed_bounds/"

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
        OUTPATH = "/Users/brucewen/Desktop/honors_thesis/estimation/combined_data/categorized_data_results_withXi/" + INPATH.split("/")[-1].split(".")[0] + "/"
        if Path(OUTPATH).is_dir()==False:
            os.mkdir(OUTPATH)
        
        # ub_v = torun[f.name.rsplit("_",1)[0]]
        ub_v=10
        print("UB_V: {}".format(ub_v))
        run_AL_profits(INPATH, OUTPATH, SELECT_RESERVE_PATH, UB_V=ub_v, num_points=100)
    else:
        continue