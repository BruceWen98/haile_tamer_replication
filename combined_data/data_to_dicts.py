import pandas as pd
import numpy as np
import pickle

data = pd.read_csv('NEW_combined_auctions_cleaned.csv')


def unique_lots(df):
    return list( (df['ID_lot']).unique())

def lot_df(lot,df):
    df = df[df["ID_lot"]==lot]
    return df

def get_estimates(lot):
    # We take the mode, in case some erros occur in certain rows/the first row.
    lows = list(lot['low_estimate'])
    highs = list(lot['high_estimate'])
    low = max(set(lows), key=lows.count)
    high = max(set(highs), key=highs.count)
    return low, high

def buyers_premium(bid, loc, auction_house):
    def adjust_bid(bid, t1, t2, r1,r2,r3):
        # t1 < middle threshold <= t2 
        # r1, r2, r3 are the rates for the three thresholds
        if bid <= t1:
            return bid*(1+r1)
        elif bid>t1 and bid <=t2:
            return t1*(1+r1) + (bid-t1)*(1+r2)
        elif bid>t2:
            return t1*(1+r1) + (t2-t1)*(1+r2) + (bid-t2)*(1+r3)
    
    if auction_house == 'christies':
        if loc == 'hong kong':
            return adjust_bid(bid, 7500000, 50000000, 0.26, 0.20, 0.145)
        elif loc == 'london':
            return adjust_bid(bid, 700000, 4500000, 0.26, 0.20, 0.145)
        elif loc == 'paris':
            return adjust_bid(bid, 700000, 4000000, 0.26, 0.20, 0.145)
        elif loc == 'new york':
            return adjust_bid(bid, 1000000, 6000000, 0.26, 0.20, 0.145)
        elif loc == 'shanghai':
            return adjust_bid(bid, 6000000, 40000000, 0.26, 0.20, 0.145)
    elif auction_house == 'sothebys':
        # only sotheby's has Overhead Premium. 1% of the bid, for "administrative costs".
        overhead_premium = 0.01 * bid
        if loc == 'hong kong':
            return adjust_bid(bid, 7500000, 40000000, 0.26, 0.20, 0.139) + overhead_premium
        elif loc == 'london':
            return adjust_bid(bid, 800000, 3800000, 0.26, 0.20, 0.139) + overhead_premium
        elif loc == 'paris':
            return adjust_bid(bid, 800000, 3500000, 0.26, 0.20, 0.139) + overhead_premium
        elif loc == 'new york':
            return adjust_bid(bid, 1000000, 4500000, 0.26, 0.20, 0.139) + overhead_premium
        elif loc == 'las vegas':
            return adjust_bid(bid, 400000, 4000000, 0.25, 0.20, 0.139) + overhead_premium
        elif loc == 'edinburgh':
            return adjust_bid(bid, 800000, 3800000, 0.25, 0.20, 0.139) + overhead_premium
        elif loc == 'monaco':
            return adjust_bid(bid, 800000, 3500000, 0.25, 0.20, 0.139) + overhead_premium
    # failure case
    print("ERROR: buyers_premium() failed to find a match for auction_house and loc.")
    print(loc, auction_house)
    return bid

def n1n_rel(lot, loc, auction_house):   ## 2nd highest bid, relative to high estimate
    _, high = get_estimates(lot)
    secondHighestBid = sorted(list(lot['bid']))[-2]
    adjustedBid = buyers_premium(secondHighestBid, loc, auction_house)
    return adjustedBid/high

def nn_rel(lot, loc, auction_house):    ## highest bid, relative to high estimate
    _, high = get_estimates(lot)
    highestBid = sorted(list(lot['bid']))[-1]
    adjustedBid = buyers_premium(highestBid, loc, auction_house)
    return adjustedBid/high

def low_rel(lot):  ## high estimate, relative to low estimate
    low, high = get_estimates(lot)
    return low/high

def num_uniq_bids(lot):
    return len(lot['bid'].unique())


def make_output_dicts(data):
    lots = unique_lots(data)
    output_dicts = []
    for lot in lots:
        lot_df_i = lot_df(lot,data)
        auctionID_lot = lot_df_i['ID_lot'].iloc[0]
        n = int(lot_df_i['N'].median())

        loc = lot_df_i['loc'].iloc[0]
        auction_house = lot_df_i['auction_house'].iloc[0]

        # Make sure not NaN
        transaction_price = nn_rel(lot_df_i, loc, auction_house)
        second_highest = n1n_rel(lot_df_i, loc, auction_house)
        low_rel_high = low_rel(lot_df_i)
        
        if not np.isnan(transaction_price) and not np.isnan(second_highest) and not np.isnan(low_rel_high):
            output_dicts.append({'auctionID_lot':auctionID_lot, 
                                    'auction_house': auction_house,
                                    '1':transaction_price, 
                                    '2':second_highest,
                                    'n':n,
                                    'loc': lot_df_i['loc'].iloc[0],
                                    'cat0': lot_df_i['cat0'].iloc[0],
                                    'cat1': lot_df_i['cat1'].iloc[0],
                                    'low_rel_high': low_rel_high,
                                    'high_estimate': get_estimates(lot_df_i)[1],
                                    'num_bids': num_uniq_bids(lot_df_i),
                                    })
        
    return output_dicts

def rem_duplicat_dicts(dicts):
    unique_auctionID_lots = np.unique([d['auctionID_lot'] for d in dicts])
    output = []
    for x in unique_auctionID_lots:
        those_xs = [d for d in dicts if d['auctionID_lot']==x]
        max_n = max([d['n'] for d in those_xs])
        corr_x = [d for d in those_xs if d['n']==max_n][0]
        output+= [corr_x]
    return output

# # remove dicts where highest bid is same as second highest bid.
# def rem_b1sameb2(dicts): 
#     out = []
#     for d in dicts:
#         if d['1']!=d['2']:
#             out += [d]
#     return out



out_dicts = make_output_dicts(data)
out_dicts = rem_duplicat_dicts(out_dicts)
# out_dicts = rem_b1sameb2(out_dicts)
print(len(out_dicts))
print(out_dicts[:10])

pickle.dump(out_dicts, open('out_dicts_ALL.p', 'wb'))