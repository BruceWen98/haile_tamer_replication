import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

output_dicts = pickle.load(open("out_dicts_ALL.p", "rb"))

## FUNCTIONS.

def filter_by_loc_cat(dicts, loc=None,cat0=None,cat1=None):
    # ## loc:
    #     - "new york"
    #     - "london"
    #     - "paris"
    #     - "shanghai"
    #     - "hong kong"
    #     - "las vegas"
    # ## cat0, cat1:
    #     - "art"
    #         - "VARIED"
    #         - "chinese art"
    #         - "impressionist/20th/21st century art"
    #         - "old masters"
    #     - "luxury goods"
    #         - "VARIED"
    #         - "jewelry"
    #     - "treasures"
    #         - "VARIED"
    #     - "wines and spirits"
    #         - "whisky"
    if loc is not None:
        dicts = [d for d in dicts if d['loc']==loc]
    if cat0 is not None:
        dicts = [d for d in dicts if d['cat0']==cat0]
    if cat1 is not None:
        dicts = [d for d in dicts if d['cat1']==cat1]

    return dicts

def filter_by_final_price(dicts, value):
    # ## value:
    #     - low: <$100,000
    #     - mid: >=$100,000, <$1,000,000
    #     - mid_high: >=$1,000,000, <$5,000,000
    #     - high: >=5M
    if value is not None:
        if value=='low':
            dicts = [d for d in dicts if d['final_price']<100000]
        elif value=='mid':
            dicts = [d for d in dicts if d['final_price']>=100000 and d['final_price']<1000000]
        elif value=='mid_high':
            dicts = [d for d in dicts if d['final_price']>=1000000 and d['final_price']<5000000]
        elif value=='high':
            dicts = [d for d in dicts if d['final_price']>=5000000]
    return dicts

def filter_by_high_est(dicts, value, pct33, pct67):
    if value is not None:
        if value=='low':
            dicts = [d for d in dicts if d['high_estimate']<=pct33]
        elif value=='mid':
            dicts = [d for d in dicts if d['high_estimate']>pct33 and d['high_estimate']<=pct67]
        elif value=='high':
            dicts = [d for d in dicts if d['high_estimate']>pct67]
    return dicts

def filter_by_high_est_MAIN(dicts, value):
    highs = [d['high_estimate'] for d in dicts if d['high_estimate']==d['high_estimate']]
    pct33, pct67 = np.percentile(highs, 33.3), np.percentile(highs, 66.7)
    return filter_by_high_est(dicts, value, pct33, pct67)

def filter_by_num_bids(dicts, degree_competition, pct33, pct67):
    if degree_competition is not None:
        if degree_competition == 'low':
            dicts = [d for d in dicts if d['num_bids']<=pct33]
        elif degree_competition == 'mid':
            dicts = [d for d in dicts if d['num_bids']>pct33 and d['num_bids']<=pct67]
        elif degree_competition == 'high':
            dicts = [d for d in dicts if d['num_bids']>pct67]
    return dicts

def filter_by_num_bids_MAIN(dicts, dicts_orig, degree_competition):
    num_bids = [d['num_bids'] for d in dicts_orig if d['num_bids']==d['num_bids']]
    pct33, pct67 = np.percentile(num_bids, 33.3), np.percentile(num_bids, 66.7)
    return filter_by_num_bids(dicts, degree_competition, pct33, pct67)

def filter_extreme_transactionprices(dicts, name_1):
    dicts = [d for d in dicts if d[name_1]==d[name_1]]    # remove nans from dicts
    initial_count = len(dicts)
    def remove_3sd(input_dicts):
        est_distribution = list([d[name_1] for d in input_dicts])
        data_mean, data_std = np.mean(est_distribution), np.std(est_distribution)
        cutOff = data_std * 3
        lower, upper = data_mean - cutOff, data_mean + cutOff
        out_dicts = []
        for d in input_dicts:
            if d[name_1] >= lower and d[name_1] <= upper:
                out_dicts.append(d)
        return out_dicts     
        
    dicts = remove_3sd(dicts)
    ns = np.unique([d['n'] for d in dicts])
    out_dicts = []
    for n in ns:
        dicts_n = [d for d in dicts if d['n']==n]
        dicts_n = remove_3sd(dicts_n)
        out_dicts.extend(dicts_n)
        
    final_count = len(out_dicts)
    print('Identified outliers: %d' % (initial_count - final_count))
    print('Non-outlier observations: %d' % final_count)
    return out_dicts
    

# Remove data that have low total number of estimates of that.
def filter_lowCounts(dicts, name_estimate, min_count):
    dicts = [d for d in dicts if d[name_estimate]==d[name_estimate]]    # remove nans from dicts
    uniq_estimates = list(set([d[name_estimate] for d in dicts]))
    estimate_dict = {}
    for est in uniq_estimates:
        length = len([d for d in dicts if d[name_estimate]==est])
        if length>=min_count:
            estimate_dict[est] = length
    out_dicts = [d for d in dicts if d[name_estimate] in estimate_dict.keys()]
    return out_dicts

# Remove auctions that have low counts of N
def filter_N(dicts, min_count):
    Ns = np.unique([d['n'] for d in dicts])
    counts_N = {}
    for n in Ns:
        count_N = len([d for d in dicts if d['n']==n])
        if count_N >= min_count:
            counts_N[n] = count_N
    
    out_dicts = [d for d in dicts if d['n'] in counts_N.keys()]
    return out_dicts
    
def filter_MAIN(dicts, loc=None, cat0=None, cat1=None, 
                value=None, degree_competition=None, name_estimate='high_estimate', 
                min_count_Estimate=5, min_count_N=5):
    
    dicts1 = filter_by_loc_cat(dicts, loc=loc, cat0=cat0, cat1=cat1)
    dicts2 = filter_by_high_est_MAIN(dicts1, value=value)
    dicts3 = filter_by_num_bids_MAIN(dicts2, dicts1, degree_competition=degree_competition)
    dicts4 = filter_extreme_transactionprices(dicts3, name_1='1')
    # dicts5 = filter_lowCounts(dicts4, name_estimate=name_estimate, min_count=min_count_Estimate)
    # dicts6 = filter_N(dicts4, min_count=min_count_N)
    return dicts4

def compute_v0(dicts):
    ratios = []
    for d in dicts:
        ratios.append(d['low_rel_high'])
    return np.median(ratios)


### MAIN. Running the code.
# Categories:
#     1) art
#     2) art, chinese art, new york
#     3) art, impressionist/20th/21st century art
#     4) art, impressionist/20th/21st century art, hong kong
#     5) art, impressionist/20th/21st century art, london
#     6) art, impressionist/20th/21st century art, new york
#     7) art, old masters
#     8) luxury goods, jewelry OR watches, new york

def make_LowMidHigh(dicts, nameOut, cat0=None, cat1=None, loc=None):
    values = ['low', 'mid', 'high']
    degree_competitions = ['low', 'mid', 'high']
    for val in values:
        d = filter_MAIN(dicts, cat0=cat0, cat1=cat1, loc=loc, value=val)
        obs = str(len(d))
        characteristics = {
            'cat0': cat0,
            'cat1': cat1,
            'loc': loc,
            'value': val,
            'degree_competition': 'All',
        }
        if d != []:
            out = {'data': d, 'characteristics': characteristics}
            pickle.dump(out, open("./categorized_data_withXi/" + nameOut+"_"+val+"_"+obs+"obs.p", "wb"))
        for deg in degree_competitions:
            d = filter_MAIN(dicts, cat0=cat0, cat1=cat1, loc=loc, value=val, degree_competition=deg)
            obs = str(len(d))
            characteristics = {
                'cat0': cat0,
                'cat1': cat1,
                'loc': loc,
                'value': val,
                'degree_competition': deg,
            }
            if d != []:
                out = {'data': d, 'characteristics': characteristics}
                pickle.dump(out, open("./categorized_data_withXi/" + nameOut+"_"+val+"_"+deg+"_"+obs+"obs.p", "wb"))
    return

def make_Aggregated(dicts, nameOut, cat0=None, cat1=None, loc=None):
    d = filter_MAIN(dicts, cat0=cat0, cat1=cat1, loc=loc)
    obs = str(len(d))
    characteristics = {
        'cat0': cat0,
        'cat1': cat1,
        'loc': loc,
        'value': None,
        'degree_competition': 'All',
    }
    if d != []:
        out = {'data': d, 'characteristics': characteristics}
        pickle.dump(out, open("./categorized_data_withXi/" + nameOut+"_"+obs+"obs.p", "wb"))
    return

make_Aggregated(output_dicts, 'ALL_d')
make_LowMidHigh(output_dicts, 'ALL')

make_Aggregated(output_dicts, 'ALL_d_art', cat0='art')
make_LowMidHigh(output_dicts, 'art', cat0='art')

make_Aggregated(output_dicts, 'ALL_d_nyc_art_chinese', cat0='art', cat1='chinese art', loc='new york')
make_LowMidHigh(output_dicts, 'art_chinese', cat0='art', cat1='chinese art', loc='new york')

make_Aggregated(output_dicts, 'ALL_d_art_imp20', cat0='art', cat1='impressionist/20th/21st century art')
make_LowMidHigh(output_dicts, 'art_imp20', cat0='art', cat1='impressionist/20th/21st century art')

make_Aggregated(output_dicts, 'ALL_d_hk_art_imp20', cat0='art', cat1='impressionist/20th/21st century art', loc='hong kong')
make_LowMidHigh(output_dicts, 'art_imp20_hk', cat0='art', cat1='impressionist/20th/21st century art', loc='hong kong')

make_Aggregated(output_dicts, 'ALL_d_ldn_art_imp20', cat0='art', cat1='impressionist/20th/21st century art', loc='london')
make_LowMidHigh(output_dicts, 'art_imp20_ldn', cat0='art', cat1='impressionist/20th/21st century art', loc='london')

make_Aggregated(output_dicts, 'ALL_d_nyc_art_imp20', cat0='art', cat1='impressionist/20th/21st century art', loc='new york')
make_LowMidHigh(output_dicts, 'art_imp20_nyc', cat0='art', cat1='impressionist/20th/21st century art', loc='new york')

make_Aggregated(output_dicts, 'ALL_d_art_oldMasters', cat0='art', cat1='old masters')
make_LowMidHigh(output_dicts, 'art_oldMasters', cat0='art', cat1='old masters')

make_Aggregated(output_dicts, 'ALL_d_lux', cat0='luxury goods')
make_LowMidHigh(output_dicts, 'lux', cat0='luxury goods')