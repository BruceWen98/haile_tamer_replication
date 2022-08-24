import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [15, 15]
import math
from collections.abc import Iterable

# # Buyer's Premium (Sotheby's)

# TODO

################################################################################
### Section 1: Auction to Lots Dataframes                                    ###
################################################################################

def add_auction_ID(df, ID):
    df['auction_ID'] = ID
    first_column = df.pop('auction_ID')
    df.insert(0, 'auction_ID', first_column)
    return df

def drop_empty_bids(df):
    df = df[df['bid'].notna()]
    return df

def unique_lots(df):
    return list(df['lot'].unique())

def frame_to_time_elapsed(df_lot, fps):
    init_frame = df_lot['time'].iloc[0]
    def diff_and_time(frame):
        diff = frame - init_frame
        time = diff/fps
        return time
    df_lot['time'] = df_lot['time'].apply(lambda x: diff_and_time(x))
    df_lot.rename(columns = {'time':'time_elapsed(s)'}, inplace = True)
    return df_lot

def time_diff(df_lot):
    time_diffs = []
    for i in range(1,df_lot.shape[0]):
        prev_time = df_lot.iloc[i-1]['time_elapsed(s)']
        curr_time = df_lot.iloc[i]['time_elapsed(s)']
        time_diffs.append(curr_time - prev_time)
    
    time_diffs = [None] + time_diffs
    df_lot['time_diff(s)'] = time_diffs
    return df_lot

def remove_duplicate_bids(df_lot):
    df_lot = df_lot.drop_duplicates(subset=["lot", "bid", 
                "estimate_currency", "low_estimate", "high_estimate"], keep='first')
    return df_lot

def lot_df(lot,df):
    df = df[df["lot"]==lot]
    return df

def drop_wrong_bids(df_lot):
    # Drop wrongly scraped bids that are less than initial value.
    init_bid = df_lot.iloc[0]['bid']
    df_lot = df_lot.drop(df_lot[df_lot.bid<init_bid].index)
    # Drop wrong bids that are below the previous bid. This sometimes happens, due to video problem.
    drop_rows = []
    for i in range(1,df_lot.shape[0]):
        if df_lot.iloc[i]['bid']<df_lot.iloc[i-1]['bid']:
            drop_rows.append(i)
    df_lot = df_lot.reset_index()
    df_lot.drop(drop_rows, axis=0, inplace=True)
    return df_lot

def df_to_lots(df):
    lots = []
    ul = unique_lots(df)
    for lot in ul:
        df_lot_orig = lot_df(lot, df)
        df_lot = remove_duplicate_bids(df_lot_orig)
        df_lot = drop_wrong_bids(df_lot)
        
        df_lot = df_lot.append(df_lot_orig.iloc[-1])
        df_lot = df_lot.reset_index()
        lots.append(df_lot)
    return lots

def money_to_int(money):
    if money is not None and type(money)==str:
        return(int(money))
    else:
        return money

def str_to_int(df):
    df['bid'] = df['bid'].apply(lambda x: money_to_int(x))
    df['low_estimate'] = df['low_estimate'].apply(lambda x: money_to_int(x))
    df['high_estimate'] = df['high_estimate'].apply(lambda x: money_to_int(x))
    return df


## Add in prev_bid_1 and prev_bid_2 columns to a lot.
def add_prev_bids(df_lot):
    prev_bid_2s = []
    prev_bid_1s = []

    for i in range(2,df_lot.shape[0]):
        prev_bid_2 = int(df_lot.iloc[i-2]['bid'])
        prev_bid_1 = int(df_lot.iloc[i-1]['bid'])
        prev_bid_2s.append(prev_bid_2)
        prev_bid_1s.append(prev_bid_1)

    prev_bid_2s = [None] + [None] + prev_bid_2s
    prev_bid_1s = [None] + [df_lot.iloc[0]['bid']] + prev_bid_1s

    df_lot["prev_bid_2"] = prev_bid_2s
    df_lot["prev_bid_1"] = prev_bid_1s
    return df_lot

def is_end(df_lot):
    df_lot['is_end'] = df_lot.apply(lambda x: x.bid == x.prev_bid_1, axis=1)
    return df_lot


## How to identify a jump bid?
# Either condition below will suffice. These are 2 types of jump bid.
# 1. The bid is increment is larger than the previous bid.
    # e.g. 1,000,000->1,100,000->1,300,000. The 2nd bid is a jump.
# 2. The bid increment is larger than the floor (from the 2nd digit onwards) of 10% of the current value.
    ## Note that this is the original Christie's bidding increment rule.

def jump_bid_type1(bid, prev_bid, prev_bid_2):
    if pd.isnull(prev_bid):
        return None
    if pd.isnull(prev_bid_2):
        return None
    bid = int(bid)
    prev_bid = int(prev_bid)
    prev_bid_2 = int(prev_bid_2)
    
    prev_jump = prev_bid - prev_bid_2
    cur_jump = bid - prev_bid

    type1 = False

    if cur_jump > prev_jump:
        jump = 1
        type1 = True

    return type1

def jump_bid_type2(bid, prev_bid):
    if pd.isnull(prev_bid):
        return None
    bid = int(bid)
    prev_bid = int(prev_bid)
    
    cur_jump = bid - prev_bid
    
    def floor_2nd_digit(x):
        len_int = len(str(x))
        order = 10**(len_int-1)
        floor = math.floor(x/order)*order/10
        return int(floor)

    type2 = False
    if cur_jump > floor_2nd_digit(prev_bid):
        type2 = True
        
    return type2


## Input: 3 bids
## Output: [(0,1), (0,1,2,3)]. 
#       First item is:
#           0: not a jump bid. 
#           1: is a jump bid.
#       Second item is:
#           0: no jump bid.
#           1: type 1 jump bid
#           2: type 2 jump bid.
#           3: both type 1 and 2 jump bid.
def jump_bid(bid, prev_bid, prev_bid_2):
    type1 = jump_bid_type1(bid, prev_bid, prev_bid_2)
    type2 = jump_bid_type2(bid, prev_bid)
    type3 = False
    jump = 0
    
    if type1 == True or type2 == True:
        jump=1
    if type1 and type2:
        type3 = True

    # Results Boolean Chain.
    if jump==0:
        return [0,0]
    if jump==1:
        if type3:
            return [1,3]
        else:
            if type1:
                return [1,1]
            if type2:
                return [1,2]

            
def add_jump_cols(df_lot):
    results = df_lot.apply(lambda x: jump_bid(x.bid, x.prev_bid_1,x.prev_bid_2), axis=1)
    df_lot['jump_bool'] = [item[0] for item in results]
    
    types = [item[1] for item in results]
    type1s = [ 1 if (i == 1 or i==3) else 0 for i in types ]
    type2s = [ 1 if (i == 2 or i==3) else 0 for i in types ]
    type3s = [ 1 if (i==3) else 0 for i in types ]
    df_lot['jump_type1'] = type1s
    df_lot['jump_type2'] = type2s
    df_lot['jump_both_type12'] = type3s
    
    return df_lot

def time_to_end(df_lot):
    max_time = max(df_lot["time_elapsed(s)"])
    time_to_end = df_lot["time_elapsed(s)"] - max_time
    df_lot["time_to_end"] = time_to_end
    return df_lot

def cdf(df_lot):
    max_bid = max(df_lot["bid"])
    cdf = df_lot["bid"] / max_bid
    df_lot["cdf"] = cdf
    return df_lot


##################################################################################################
##### FINAL FUNCTION TO USE.
def auction_to_lots(auction_df):
    auction_df = str_to_int(auction_df)
    lots = df_to_lots(auction_df)
    out_lot_dfs = []
    for lot in lots:
        lot = frame_to_time_elapsed(lot, 30)
        lot = time_diff(lot)
        lot = add_prev_bids(lot)
        lot = is_end(lot)
        lot = add_jump_cols(lot)
        lot = lot.iloc[: , 2:]
        lot = time_to_end(lot)
        lot = cdf(lot)
        
        ## We need to remove lots that have problems, i.e. for some reason bidding goes down at some point.
        # Check that bidding is increasing.
        okay = True
        for i in range(lot.shape[0]-1):
            if lot['bid'][i+1] < lot['bid'][i]:
                okay = False
                break
        if okay==True:
            out_lot_dfs.append(lot)
        
    return out_lot_dfs




################################################################################
### Section 2: Plotting Lots                                                 ###
################################################################################

def plot_bids(lot,ax):
    def get_estimates(lot):
        # We take the mode, in case some erros occur in certain rows/the first row.
        lows = list(lot['low_estimate'])
        highs = list(lot['high_estimate'])
        low = max(set(lows), key=lows.count)
        high = max(set(highs), key=highs.count)
        return low, high
    
    low, high = get_estimates(lot)
    mid = np.mean([high,low])
    
    normalized_bid = lot['bid']/mid
    lot['norm_bid'] = normalized_bid
    ax.plot(lot['time_elapsed(s)'],lot['norm_bid'])
    return

def plot_all_auction(auction, auction_title, ax):
    out_lot_dfs = auction_to_lots(auction)
    for lot in out_lot_dfs:
        plot_bids(lot, ax)
        
    ax.set_ylabel('Bid relative to Middle Estimate')
    ax.set_xlabel('Seconds Elapsed')
    ax.title.set_text(auction_title)
    return

# CDF plotting
def plot_cdf(lot,ax):
    ax.plot(lot['time_to_end'],lot['cdf'])
    return

def plot_all_cdf(auction, auction_title, ax):
    out_lot_dfs = auction_to_lots(auction)
    for lot in out_lot_dfs:
        plot_cdf(lot, ax)
        
    ax.set_ylabel('Bid relative to Final Bid')
    ax.set_xlabel('Seconds to End')
    ax.title.set_text(auction_title)
    return

################################################################################
### Section 3: Lot Statistics                                                ###
################################################################################
# # Lot Statistics
# This section will extract statistics from each lot.

# We will be interested in the following statistics:

# - total number of bids
# - Overall frequency of bidding (How to define this?)
#     - number of bids / total time elapsed
# - number of jump bids
#     - number of type 1 jump bids
#     - number of type 2 jump bids
#     - number of both type 1 & 2 jump bids
# - frequency of jump bids
#     - all types
#     - type 1
#     - type 2
#     - type 1 and type 2
# - initial bid relative to estimate
# - final bid relative to estimate
# - final time taken to conclude auction. (i.e. last time_diff(s))

# - Time-Segmented Statistics:
#     - We will split into N (e.g. 10) sub-spaces of times. For each:
#         - number of bids relative to total
#         - number of jump bids relative to total
#             - type1
#             - type2
#         - final price of that time segment relative to final price in the end
#         - frequency of bidding in the sub-space
#             - number of bids / time elapsed in the sub-space
#         - "Bid Gradient": What is the price rise in this time?
#             - Intuition: Price Rise / Time Difference

def total_bids(lot):
    return len(lot['bid'])-1

def frequency_bids(lot):
    total_time_lot = max(lot["time_elapsed(s)"])
    if total_time_lot == 0:
        return 0.0
    return total_bids(lot)/total_time_lot

def num_jump_bids(lot):
    return sum(lot['jump_bool'] == 1)

def num_jump1(lot):
    return sum(lot['jump_type1'] == 1)

def num_jump2(lot):
    return sum(lot['jump_type2'] == 1)

def num_jump12(lot):
    return sum(lot['jump_both_type12'] == 1)

# Frequency of Jump Bids
def freq_jump_bids(lot):
    total_time_lot = max(lot["time_elapsed(s)"])
    if total_time_lot == 0:
        return 0.0,0.0,0.0,0.0
    freq_jump = num_jump_bids(lot)/total_time_lot
    freq_jump1 = num_jump1(lot)/total_time_lot
    freq_jump2 = num_jump2(lot)/total_time_lot
    freq_jump12 = num_jump12(lot)/total_time_lot
    return freq_jump,freq_jump1,freq_jump2,freq_jump12

def get_estimates(lot):
    # We take the mode, in case some erros occur in certain rows/the first row.
    lows = list(lot['low_estimate'])
    highs = list(lot['high_estimate'])
    low = max(set(lows), key=lows.count)
    high = max(set(highs), key=highs.count)
    return low, high

def init_rel(lot):
    low, high = get_estimates(lot)
    mid = (high+low)/2
    init_bid = list(lot['bid'])[0]
    return init_bid/mid

def final_rel(lot):
    low, high = get_estimates(lot)
    mid = (high+low)/2
    final_bid = list(lot['bid'])[-1]
    return final_bid/mid

def time_conclude(lot):
    return list(lot['time_diff(s)'])[-1]



## Time Segmented Statistics
# Return N Segments of the Total Time.
# e.g. N=10, total_time_lot=300s, then, the segments will be:
#     30,60,90,...,300
def segment_times(lot, N):
    total_time_lot = max(lot["time_elapsed(s)"])
    return np.linspace(0, total_time_lot, N+1)

def get_segment_values(low_time, high_time, lot):
    segmented_lot = lot[(lot["time_elapsed(s)"]>=low_time) & (lot["time_elapsed(s)"]<=high_time)]

    n_bids = total_bids(segmented_lot)
    n_jump_bids = num_jump_bids(segmented_lot)
    n_jump1 = num_jump1(segmented_lot)
    n_jump2 = num_jump2(segmented_lot)
    n_jump12 = num_jump12(segmented_lot)

    low, high = get_estimates(lot)
    mid = (high+low)/2
    if high_time-low_time == 0:
        freq_bidding_segment = 0.0
    else:
        freq_bidding_segment = n_bids/(high_time-low_time)
    try:
        final_price_segment = list(segmented_lot['bid'])[-1]
        final_rel_price_segment = final_price_segment/mid
        if high_time-low_time == 0:
            bid_gradient = 0.0
        else:
            bid_gradient = (final_price_segment - list(segmented_lot['bid'])[0]) / (high_time-low_time) / mid
    except:
        final_price_segment = np.nan
        final_rel_price_segment = np.nan
        bid_gradient = np.nan
    
    
    return [
        n_bids,
        n_jump_bids,
        n_jump1,
        n_jump2,
        n_jump12,
        final_price_segment,
        final_rel_price_segment,
        freq_bidding_segment,
        bid_gradient
    ]

def segment_stats(lot, N):
    segmented_times = segment_times(lot, N)
    out_stats = []
    for i in range(N):
        low_time = segmented_times[i]
        high_time = segmented_times[i+1]
        segment_values = get_segment_values(low_time, high_time, lot)
        out_stats.append(segment_values)
    return out_stats


## Get All Lot Statistics
def get_all_lot_stats(lot, N_segments):
    freq_jump,freq_jump1,freq_jump2,freq_jump12 = freq_jump_bids(lot)
    segmented_stats = segment_stats(lot, N_segments)
    out_list = [
        total_bids(lot),
        frequency_bids(lot),
        num_jump_bids(lot),
        num_jump1(lot),
        num_jump2(lot),
        num_jump12(lot),
        freq_jump,
        freq_jump1,
        freq_jump2,
        freq_jump12,
        init_rel(lot),
        final_rel(lot),
        time_conclude(lot),
        segmented_stats
    ]
    
    return out_list


def make_col_names(lot_stats):
    N_segments = len(lot_stats[-1])
    col_names = [
        "total_bids",
        "overall_freq",
        "num_jump",
        "num_jump1",
        "num_jump2",
        "num_jump12",
        "freq_jump",
        "freq_jump1",
        "freq_jump2",
        "freq_jump12",
        "init_rel",
        "final_rel",
        "last_time",
    ]
    for i in range(N_segments):
        segment_list = [
            "seg%d_bids" % i,
            "seg%d_jump_bids" % i,
            "seg%d_jump1" % i,
            "seg%d_jump2" % i,
            "seg%d_jump12" % i,
            "seg%d_final_price" % i,
            "seg%d_final_rel_price" % i,
            "seg%d_freq_bid" % i,
            "seg%d_bid_gradient" % i
        ]
        col_names = col_names + segment_list
        
    return col_names


def flatten(l):
    for el in l:
        if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el

def make_stats_df_auction(lots, N_segments, auction):
    auction_ID = auction['auction_ID'][0]
    list_dicts = []
    for lot in lots:
        lot_stats = get_all_lot_stats(lot, N_segments)
        flat_list = list(flatten(lot_stats))
        col_names = make_col_names(lot_stats)
        col_names = ["auction_ID","lot","lot_name"] + col_names
        
        lot_num = lot['lot'][0]
        lot_name = lot['lot_name'][0]
        lot_stats = [auction_ID,lot_num,lot_name] + flat_list
        
        dict_lot = dict(zip(col_names, lot_stats))
        list_dicts.append(dict_lot)

    return pd.DataFrame(list_dicts)    




###############################################################################
### Section 4: Overall Processing                                           ###
###############################################################################

def process_1_sotheby_auction(csv_path, auction_ID, N_segments):
    auction = pd.read_csv(csv_path)
    auction = add_auction_ID(auction, auction_ID)
    auction = drop_empty_bids(auction)
    lots = auction_to_lots(auction)
    
    stats_df = make_stats_df_auction(lots, N_segments, auction)
    return auction, lots, stats_df
