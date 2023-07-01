import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt


data = pickle.load(open('/Users/brucewen/Desktop/honors_thesis/estimation/combined_data/out_dicts_ALL.p', 'rb'))
df = pd.DataFrame(data)

df.to_csv('/Users/brucewen/Desktop/honors_thesis/estimation/combined_data/out_dicts_ALL.csv')

# remove rows where n is larger than num_bids
df = df[df['n'] <= df['num_bids']]

n_lb = df['n']
n_ub = df['num_bids']

diffs = n_ub - n_lb

sns.set_style('darkgrid')
plt.figure(figsize=(10,6), tight_layout=True)
plt.title("Distribution of n_bar and n_under (Old Data)")
plt.xlabel("number of bidders")
plt.ylabel("count")
plt.hist(diffs, bins=100,)

plt.legend()
plt.show()