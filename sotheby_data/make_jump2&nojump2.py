# Make List of Auction Global-ID's that contain Jump Type 2.
import pandas as pd
import pickle
from tqdm import tqdm

def makeGlobalID(df):
    df['global_ID'] = df['auction_ID'] + df['lot'].astype(str) + df['lot_name']
    return df

df = pd.read_csv("sothebys_stats.csv")
df = makeGlobalID(df)

df_jump2 = df[df['num_jump1'] > 0]
df_no_jump2 = df[df['num_jump1'] == 0]

list_jump2 = df_jump2['global_ID'].unique()
list_no_jump2 = df_no_jump2['global_ID'].unique()



def makeBidRel(df):
    df['bid_rel'] = df['bid'] / ( (df['low_estimate']+df['high_estimate'])/2 )
    return df

dataset = pd.read_csv("sotheby_ALL.csv")
dataset = makeGlobalID(dataset)
dataset = makeBidRel(dataset)

# Jump2 Max 2 Bids
dataset_jump2 = dataset[dataset['global_ID'].isin(list_jump2)]
IDs = dataset_jump2['global_ID'].unique()
output_dicts_jump2 = []
for id in tqdm(IDs):
    df_id = dataset_jump2[dataset_jump2['global_ID'] == id]
    df_id = df_id.sort_values(by='bid_rel', ascending=False).reset_index()
    if df_id.shape[0] < 2:
        continue
    top1_bid = df_id.loc[0, 'bid_rel']
    top2_bid = df_id.loc[1, 'bid_rel']
    output_dict = {}
    output_dict['1'] = top1_bid
    output_dict['2'] = top2_bid
    output_dicts_jump2.append(output_dict)

with open('output_dicts_jump2.pickle', 'wb') as handle:
    pickle.dump(output_dicts_jump2, handle, protocol=pickle.HIGHEST_PROTOCOL)

# No Jump2 Max 2 Bids
dataset_no_jump2 = dataset[dataset['global_ID'].isin(list_no_jump2)]
IDs = dataset_no_jump2['global_ID'].unique()
output_dicts_NOjump2 = []
for id in tqdm(IDs):
    df_id = dataset_no_jump2[dataset_no_jump2['global_ID'] == id]
    df_id = df_id.sort_values(by='bid_rel', ascending=False).reset_index()
    if df_id.shape[0] < 2:
        continue
    top1_bid = df_id.loc[0, 'bid_rel']
    top2_bid = df_id.loc[1, 'bid_rel']
    output_dict = {}
    output_dict['1'] = top1_bid
    output_dict['2'] = top2_bid
    output_dicts_NOjump2.append(output_dict)


with open('output_dicts_NOjump2.pickle', 'wb') as handle:
    pickle.dump(output_dicts_NOjump2, handle, protocol=pickle.HIGHEST_PROTOCOL)