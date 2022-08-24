import pandas as pd
from tqdm import tqdm
import pickle

# Functions to Turn Lot dataframe into Panel Form with T periods.

def makeGlobalID(df):
    df['global_ID'] = df['auction_ID'] + df['lot'].astype(str) + df['lot_name']
    return df

def makeBidRel(df):
    df['bid_rel'] = df['bid'] / ( (df['low_estimate']+df['high_estimate'])/2 )
    return df

dataset = pd.read_csv("sotheby_ALL.csv")
dataset = makeGlobalID(dataset)
dataset = makeBidRel(dataset)

IDs = dataset['global_ID'].unique()
output_dicts = []
for id in tqdm(IDs):
    df_id = dataset[dataset['global_ID'] == id]
    df_id = df_id.sort_values(by='bid_rel', ascending=False).reset_index()
    if df_id.shape[0] < 2:
        continue
    top1_bid = df_id.loc[0, 'bid_rel']
    top2_bid = df_id.loc[1, 'bid_rel']
    output_dict = {}
    output_dict['1'] = top1_bid
    output_dict['2'] = top2_bid
    output_dicts.append(output_dict)

with open('top2_bids.pickle', 'wb') as handle:
    pickle.dump(output_dicts, handle, protocol=pickle.HIGHEST_PROTOCOL)

