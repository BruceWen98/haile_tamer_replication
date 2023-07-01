import numpy as np
import pandas as pd
import pickle

## Turns data into top2 bid dicts with N and category.
data = pd.read_csv('combined_auctions.csv')
n_data1 = pd.read_csv("bidders_ALL35sotheby_v2.csv")
n_data2 = pd.read_csv("bidders_GCSR_sotheby.csv")

def unique_lots(df):
    return list( (df['auction_ID'] + "_" + df['lot'].apply(str) ).unique())

def lot_df(lot,df):
    df = df[df["auctionID_lot"]==lot]
    return df

def get_estimates(lot):
    # We take the mode, in case some erros occur in certain rows/the first row.
    lows = list(lot['low_estimate'])
    highs = list(lot['high_estimate'])
    low = max(set(lows), key=lows.count)
    high = max(set(highs), key=highs.count)
    return low, high

def n1n_rel(lot):   ## 2nd highest bid, relative to low estimate
    _, high = get_estimates(lot)
    secondHighestBid = sorted(list(lot['bid']))[-2]
    return secondHighestBid/high

def nn_rel(lot):    ## highest bid, relative to low estimate
    _, high = get_estimates(lot)
    highestBid = sorted(list(lot['bid']))[-1]
    return highestBid/high

def final_price(lot):
    return sorted(list(lot['bid']))[-1]

def low_rel(lot):  ## high estimate, relative to low estimate
    low, high = get_estimates(lot)
    return low/high

def num_uniq_bids(lot):
    return len(lot['bid'].unique())

def match_N(data, n_data):
    data['auctionID_lot'] = data['auction_ID'] + "_" + data['lot'].apply(str)
    lots = unique_lots(data)
    n_data['auctionID_lot'] = n_data['auction_ID'] + '_' + n_data['lot'].apply(str)
    output_dicts = []
    for lot in lots:
        lot_df_i = lot_df(lot,data)
        auctionID_lot = lot_df_i['auctionID_lot'].iloc[0]
        if auctionID_lot in list(n_data['auctionID_lot']):
            n = n_data[n_data['auctionID_lot']==auctionID_lot]['num_uniq_bidders'].iloc[0]
            if num_uniq_bids(lot_df_i)<2:
                continue
            elif n<2 and num_uniq_bids(lot_df_i)>=2:
                n=2
            output_dicts.append({'auctionID_lot':auctionID_lot, 
                                 '1':nn_rel(lot_df_i), 
                                 '2':n1n_rel(lot_df_i),
                                 'n':n,
                                 'loc': lot_df_i['loc'].iloc[0],
                                 'cat0': lot_df_i['cat0'].iloc[0],
                                 'cat1': lot_df_i['cat1'].iloc[0],
                                 'low_rel_high': low_rel(lot_df_i),
                                 'final_price': final_price(lot_df_i),
                                 'high_estimate': get_estimates(lot_df_i)[1],
                                 })
        else:
            continue
        
    return output_dicts

output_dicts1 = match_N(data, n_data1)
output_dicts2 = match_N(data, n_data2)

output_dicts = output_dicts1 + output_dicts2
print(len(output_dicts))

uniqs = np.unique([d['auctionID_lot'] for d in output_dicts])
print(len(uniqs))

def rem_duplicat_dicts(dicts):
    unique_auctionID_lots = np.unique([d['auctionID_lot'] for d in dicts])
    output = []
    for x in unique_auctionID_lots:
        those_xs = [d for d in dicts if d['auctionID_lot']==x]
        max_n = max([d['n'] for d in those_xs])
        corr_x = [d for d in those_xs if d['n']==max_n][0]
        output+= [corr_x]
    return output


output_dicts = rem_duplicat_dicts(output_dicts)
pickle.dump(output_dicts, open("output_dicts_sotheby_cats_N.p", "wb"))


