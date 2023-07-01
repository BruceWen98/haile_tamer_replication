import pickle
import numpy as np
import pandas as pd

IN_BOUNDS_PATHS = [
    # '/Users/brucewen/Desktop/honors_thesis/estimation/combined_data/categorized_data_results_withXi/ALL_d_art_1712obs/allbounds_vN.p',
    # '/Users/brucewen/Desktop/honors_thesis/estimation/combined_data/categorized_data_results_withXi/ALL_d_art_imp20_1477obs/allbounds_vN.p',
    '/Users/brucewen/Desktop/honors_thesis/estimation/combined_data/categorized_data_results_withXi/ALL_d_nyc_art_imp20_497obs/bounds_vN_n3.p',
    '/Users/brucewen/Desktop/honors_thesis/estimation/combined_data/categorized_data_results_withXi/ALL_d_nyc_art_imp20_497obs/bounds_vN_n4.p',
    # '/Users/brucewen/Desktop/honors_thesis/estimation/combined_data/categorized_data_results_withXi/art_imp20_nyc_mid_mid_48obs/allbounds_vN.p',
#     '/Users/brucewen/Desktop/honors_thesis/estimation/combined_data/categorized_data_results_withXi/ALL_d_art_oldMasters_91obs/allbounds_vN.p',
#     '/Users/brucewen/Desktop/honors_thesis/estimation/combined_data/categorized_data_results_withXi/ALL_d_lux_308obs/allbounds_vN.p',
]

IN_DICTS_PATHS = [
    # '/Users/brucewen/Desktop/honors_thesis/estimation/combined_data/categorized_data_withXi/ALL_d_art_1712obs.p',
    # '/Users/brucewen/Desktop/honors_thesis/estimation/combined_data/categorized_data_withXi/ALL_d_art_imp20_1477obs.p',
    '/Users/brucewen/Desktop/honors_thesis/estimation/combined_data/categorized_data_withXi/ALL_d_nyc_art_imp20_497obs.p',
    '/Users/brucewen/Desktop/honors_thesis/estimation/combined_data/categorized_data_withXi/ALL_d_nyc_art_imp20_497obs.p',
    # '/Users/brucewen/Desktop/honors_thesis/estimation/combined_data/categorized_data_withXi/art_imp20_nyc_mid_mid_48obs.p',
    # '/Users/brucewen/Desktop/honors_thesis/estimation/combined_data/categorized_data_withXi/ALL_d_art_oldMasters_91obs.p',
    # '/Users/brucewen/Desktop/honors_thesis/estimation/combined_data/categorized_data_withXi/ALL_d_lux_308obs.p',
]


## Helping Functions
def get_interval(X,bounds_dicts):
    out = {}
    for bounds_dict in bounds_dicts:
        n = bounds_dict['N']
        max_lb = float('-inf')
        max_lb_index = -1
        
        lbs = bounds_dict['lb']
        ubs = bounds_dict['ub']
        max_piL = max(lbs)
        
        for i, lb in enumerate(lbs):
            if lb > max_lb:
                max_lb = lb
                max_lb_index = i

        j=0
        pL_idx = 0
        while j < max_lb_index:
            if ubs[j] <= max_piL:
                pL_idx = j
            j+=1
            
        pL = X[pL_idx]
        
        j = max_lb_index
        pU_idx = j
        while j<len(X):
            if ubs[j] <= max_piL:
                pU_idx = j
                break
            j+=1
        pU = X[pU_idx]
        
        out[n] = (pL, pU)
        
    return out

def get_argmax_lb(X, bounds_dicts):
    out = {}
    for bounds_dict in bounds_dicts:
        n = bounds_dict['N']
        max_lb = float('-inf')
        max_lb_index = -1
        
        lbs = bounds_dict['lb']
        
        for i, lb in enumerate(lbs):
            if lb > max_lb:
                max_lb = lb
                max_lb_index = i

        out[n] = X[max_lb_index]
    return out

def max_argmax_lb(bounds_dicts):
    out = {}
    for bounds_dict in bounds_dicts:
        n = bounds_dict['N']
        max_lb = float('-inf')
        max_lb_index = -1
        
        lbs = bounds_dict['lb']
        
        for i, lb in enumerate(lbs):
            if lb > max_lb:
                max_lb = lb
                max_lb_index = i

        ubs = bounds_dict['ub']
        ubs_CI95 = bounds_dict['ub_95']
        out[n] = ubs[max_lb_index], ubs_CI95[max_lb_index]
    return out

def get_max_lb(bounds_dicts):
    out = {}
    for bounds_dict in bounds_dicts:
        n = bounds_dict['N']
        lbs = bounds_dict['lb']
        lbs_CI95 = bounds_dict['lb_95']
        out[n] = max(lbs), lbs_CI95[lbs.index(max(lbs))]
    return out

def get_v0_profit_ub(bounds_dicts):
    out = {}
    for bounds_dict in bounds_dicts:
        n = bounds_dict['N']
        ubs = bounds_dict['ub']
        ubs_CI95 = bounds_dict['ub_95']
        out[n] = ubs[0], ubs_CI95[0]
    return out

def get_v0_profit_lb(bounds_dicts):
    out = {}
    for bounds_dict in bounds_dicts:
        n = bounds_dict['N']
        lbs = bounds_dict['lb']
        lbs_CI95 = bounds_dict['lb_95']
        out[n] = lbs[0], lbs_CI95[0]
    return out

def min_pi_increases(X, bounds_dicts):
    out_v0 = get_v0_profit_ub(bounds_dicts)
    out_est = get_max_lb(bounds_dicts)
    # minus the two dictionaries
    out = {}
    for n in out_v0:
        out[n] = out_est[n][0] - out_v0[n][0], out_est[n][1] - out_v0[n][1]
    return out

def max_pi_increases(X, bounds_dicts):
    out_v0 = get_v0_profit_lb(bounds_dicts)
    out_est = max_argmax_lb(bounds_dicts)
    # minus the two dictionaries
    out = {}
    for n in out_v0:
        out[n] = out_est[n][0] - out_v0[n][0], out_est[n][1] - out_v0[n][1]
    return out

def convert_to_usd(amount, loc):
    exchange_rates = {'hong kong': 0.13, 
                      'paris': 1.11, 
                      'london': 1.25, 
                      'shanghai': 0.15,
                      'new york': 1.00,
                      'las vegas': 1.00,
                      'edinburgh': 1.25,
                      'monaco': 1.11,
    }
    usd_amount = amount * exchange_rates[loc]
    return usd_amount

def get_average_high_est(dicts, N=None):
    if N is not None:
        dicts = [d for d in dicts if d['n']==N]
        
    for d in dicts:
        d['high_estimate'] = convert_to_usd(d['high_estimate'], d['loc'])
    return np.mean([d['high_estimate'] for d in dicts])
    
def min_expected_profit_increase(dicts, X, bounds_dicts):
    pi_increases_dict = min_pi_increases(X, bounds_dicts)
    Ns = np.unique([d['n'] for d in dicts])
    out = {}
    for n in Ns:
        try:
            avg_high_est = get_average_high_est(dicts, N=n)
            pi_increase, pi_increase_CI95 = pi_increases_dict[n]
            out[n] = pi_increase * avg_high_est, pi_increase_CI95 * avg_high_est
        except KeyError:
            pass
    
    return out

def max_expected_profit_increase(dicts, X, bounds_dicts):
    pi_increases_dict = max_pi_increases(X, bounds_dicts)
    Ns = np.unique([d['n'] for d in dicts])
    out = {}
    for n in Ns:
        try:
            avg_high_est = get_average_high_est(dicts, N=n)
            pi_increase, pi_increase_CI95 = pi_increases_dict[n]
            out[n] = pi_increase * avg_high_est, pi_increase_CI95 * avg_high_est
        except KeyError:
            pass
    
    return out

def get_avg_transaction_price(dicts):
    Ns = np.unique([d['n'] for d in dicts])
    out = {}
    for n in Ns:
        n_dicts = [d for d in dicts if d['n']==n]
        if n < max(Ns):
            try:
                high_est_avg = get_average_high_est(n_dicts)
                out[n] = np.mean([d['1'] for d in n_dicts]) * high_est_avg
            except KeyError:
                pass
    print(out)
    return out

# Make these quantities into a df with columns: N, choice, min_expected_profit_increase, max_expected_profit_increase
def make_df(X, bounds_dicts, data_dicts, selected_N = [3,4]):
    choices = get_argmax_lb(X, bounds_dicts)
    min_increases = min_expected_profit_increase(data_dicts, X, bounds_dicts)
    max_increases = max_expected_profit_increase(data_dicts, X, bounds_dicts)
    intervals = get_interval(X, bounds_dicts)
    avg_prices = get_avg_transaction_price(data_dicts)

    # Filter to selected_N
    avg_prices = {k: v for k, v in avg_prices.items() if k in selected_N}
    print(choices)

    df = pd.DataFrame({'N': choices.keys(), 
                       'min_interval': [x[0] for x in intervals.values()],
                       'max_interval': [x[1] for x in intervals.values()],
                       'choice': choices.values(), 
                       'Average Price': avg_prices.values(),
                       'min_expected_profit_increase': min_increases.values(),
                       'max_expected_profit_increase': max_increases.values(),
                       })
    
    # filter df to only selected_N
    df = df[df['N'].isin(selected_N)]
    return df.T


OUTPATH = "/Users/brucewen/Desktop/honors_thesis/estimation/combined_data/select_reserve/"
for i in range(len(IN_BOUNDS_PATHS)):
    IN_BOUNDS_PATH = IN_BOUNDS_PATHS[i]
    IN_DICTS_PATH = IN_DICTS_PATHS[i]
    with open(IN_BOUNDS_PATH, 'rb') as f:
        [X, bounds_dicts] = pickle.load(f)
    with open(IN_DICTS_PATH, 'rb') as f:
        data_dicts = pickle.load(f)['data']
        
    make_df(X, bounds_dicts, data_dicts, selected_N = [3]).to_csv(OUTPATH + IN_DICTS_PATHS[i].split("/")[-1].split(".")[0] + ".csv")