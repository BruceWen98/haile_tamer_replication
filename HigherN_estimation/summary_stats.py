import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt


data = pickle.load(open("out_dicts_ALL.p", "rb"))
df = pd.DataFrame(data)

def get_MedianMeanStdMinMax(arr):
    return np.median(arr), np.mean(arr), np.std(arr), min(arr), max(arr)

### SUMMARY STATISTICS ###

def summary_statistics(data):
    # 1. Transaction price, i.e. nn_rel
    tpMedian, tpMean, tpStd, tpMin, tpMax = get_MedianMeanStdMinMax(list(data['1']))
    print("Transaction price. Median: {}, Mean: {}, Std: {}, Min: {}, Max: {}".format(tpMedian, tpMean, tpStd, tpMin, tpMax))

    # 2. 2nd-Highest bid, i.e. n1n_rel
    h2Median, h2Mean, h2Std, h2Min, h2Max = get_MedianMeanStdMinMax(data['2'])
    print("2nd-Highest bid. Median: {}, Mean: {}, Std: {}, Min: {}, Max: {}".format(h2Median, h2Mean, h2Std, h2Min, h2Max))

    # 3. Number of Bidders, i.e. N
    NMedian, NMean, NStd, NMin, NMax = get_MedianMeanStdMinMax(data['n'])
    print("Number of bidders. Median: {}, Mean: {}, Std: {}, Min: {}, Max: {}".format(NMedian, NMean, NStd, NMin, NMax))

    # 4. Number of auction lots
    N_ID_lots = data.shape[0]
    print("Number of auction lots: {}".format(N_ID_lots))

    # 5. Low Estimate Relative to High Estimate
    lhMedian, lhMean, lhStd, lhMin, lhMax = get_MedianMeanStdMinMax(data['low_rel_high'])
    print("Low estimate relative to high estimate. Median: {}, Mean: {}, Std: {}, Min: {}, Max: {}".format(lhMedian, lhMean, lhStd, lhMin, lhMax))
    
    # 6. High Estimate
    hMedian, hMean, hStd, hMin, hMax = get_MedianMeanStdMinMax(data['high_estimate'])
    print("High estimate. Median: {}, Mean: {}, Std: {}, Min: {}, Max: {}".format(hMedian, hMean, hStd, hMin, hMax))
    
    # 7. Number of Bids
    bNMedian, bNMean, bNStd, bNMin, bNMax = get_MedianMeanStdMinMax(data['num_bids'])
    print("Number of bids. Median: {}, Mean: {}, Std: {}, Min: {}, Max: {}".format(bNMedian, bNMean, bNStd, bNMin, bNMax))
    
    return {
        'Transaction Price': [tpMedian, tpMean, tpStd, tpMin, tpMax],
        '2nd-Highest Bid': [h2Median, h2Mean, h2Std, h2Min, h2Max],
        'Number of Bidders': [NMedian, NMean, NStd, NMin, NMax],
        'Number of Auction Lots': N_ID_lots,
        'Low Estimate Relative to High Estimate': [lhMedian, lhMean, lhStd, lhMin, lhMax],
        'Number of Bids': [bNMedian, bNMean, bNStd, bNMin, bNMax],
    }
    
data_C = df[df['auction_house']=='christies']
data_S = df[df['auction_house']=='sothebys']
summary_statistics(data_C)
summary_statistics(data_S)
summary_statistics(df)

# Transaction price. Median: 1.3316666666666666, Mean: 2.1358945898782915, Std: 3.8844803937965455, Min: 0.448, Max: 76.375
# 2nd-Highest bid. Median: 1.26, Mean: 1.9561541555818822, Std: 3.6005566956514206, Min: 0.42, Max: 75.8025
# Number of bidders. Median: 6.0, Mean: 6.453030303030303, Std: 3.3852496303356663, Min: 2, Max: 23
# Number of auction lots: 660
# Low estimate relative to high estimate. Median: 0.6666666666666666, Mean: 0.668423562557435, Std: 0.06638632301183733, Min: 0.44, Max: 1.0
# High estimate. Median: 1000000.0, Mean: 4858380.303030303, Std: 14818382.30403036, Min: 2000.0, Max: 280000000.0
# Number of bids. Median: 11.0, Mean: 12.722727272727273, Std: 8.774814086859797, Min: 1, Max: 64

# Transaction price. Median: 1.3546666666666667, Mean: 2.2199029736582068, Std: 4.370219172784521, Min: 0.33866666666666667, Max: 134.4
# 2nd-Highest bid. Median: 1.27, Mean: 1.9639118822812047, Std: 3.7919731851141365, Min: 0.08466666666666667, Max: 126.0
# Number of bidders. Median: 5.0, Mean: 5.7051490514905145, Std: 3.2344354071858756, Min: 2, Max: 26
# Number of auction lots: 1845
# Low estimate relative to high estimate. Median: 0.6666666666666666, Mean: 0.6712120779211262, Std: 0.0730159032915572, Min: 0.3333333333333333, Max: 0.8888888888888888
# High estimate. Median: 350000.0, Mean: 4086008.075880759, Std: 14826686.490636896, Min: 500.0, Max: 230000000.0
# Number of bids. Median: 9.0, Mean: 11.359349593495935, Std: 8.54742076564582, Min: 1, Max: 88



### Distribution of 2nd Highest Bid and Transaction Price ###
B2 = df['2']
B1 = df['1']

# Remove outliers
B2 = B2[B2 < 10]
B1 = B1[B1 < 10]

sns.set_style('darkgrid')
plt.figure(figsize=(10,6), tight_layout=True)
plt.title("Distribution of 2nd Highest Bid and Transaction Price")
plt.xlabel("Relative Price to High Estimate")
plt.ylabel("count")
plt.hist(B1, bins=100, alpha=0.5, label='Transaction Price')
plt.hist(B2, bins=100, alpha=0.5, label='2nd Highest Bid')

plt.legend()
plt.savefig('2nd_Highest_Bid_Transaction_Price.png')

# loop over each N and plot the distribution of 2nd highest bid and transaction price
for n in range(2, max(df['n'])+1):
    B2 = df[df['n'] >= n]['2']
    B1 = df[df['n'] >= n]['1']
    
    # Remove outliers
    B2 = B2[B2 < 10]
    B1 = B1[B1 < 10]
    
    sns.set_style('darkgrid')
    plt.figure(figsize=(10,6), tight_layout=True)
    plt.title(f"Distribution of 2nd Highest Bid and Transaction Price for all $\hat{{N}}$ >= {n}")
    plt.xlabel("Relative Price to High Estimate")
    plt.ylabel("count")
    plt.hist(B1, bins=100, alpha=0.5, label='Transaction Price')
    plt.hist(B2, bins=100, alpha=0.5, label='2nd Highest Bid')

    plt.legend()
    plt.savefig("/Users/brucewen/Desktop/honors_thesis/estimation/HigherN_estimation/dist_2nd_1st_bids/" +
                '2nd_Highest_Bid_Transaction_Price_N_geq{}.png'.format(n))


### Distribution of N, the number of bidders ###
ns = df['n']
ns = ns[ns < 30]

sns.set_style('darkgrid')
plt.figure(figsize=(10,6), tight_layout=True)
# Set font to 20
plt.rc('font', size=16)
plt.title("Distribution of Estimated Numbers of Bidders")
plt.xlabel(r"$N$")
plt.ylabel("Count")
# hist with spacing for every integer
plt.hist(ns, bins=np.arange(2, 30, 1), alpha=0.5, label=r"$N$")

plt.legend()
plt.savefig('Distribution_N.png')


### Distribution of N relative to the number of bids ###
num_bids = df['num_bids']
diff = num_bids - ns
sns.set_style('darkgrid')
plt.figure(figsize=(10,6), tight_layout=True)
plt.title(r"Distribution of Difference between Number of Bids and $\hat{N}$")
plt.xlabel(r"Number of Bids $-$ $\hat{N}$")
plt.ylabel("count")
# hist with spacing for every integer
plt.hist(diff, bins=np.arange(1, 30, 1), alpha=0.5, label=r"numBids$-\hat{N}$")

plt.legend()
plt.savefig('Distribution_diff_numBids_N.png')



## Distribution of Categories
counts = df.fillna('N/A').groupby(['cat0','cat1','loc']).size().reset_index(name='count')
print(counts)
print(df.shape[0])