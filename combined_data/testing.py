import pickle
import matplotlib.pyplot as plt
import numpy as np

output_dicts = pickle.load(open("out_dicts_ALL.p", "rb"))

# highs = [d['high_estimate'] for d in output_dicts if d['high_estimate']==d['high_estimate']]
# print(highs)
# print(np.percentile(highs, 33.3))
# print(np.percentile(highs, 66.7))
# print(np.percentile(highs, 100))



# ds = pickle.load(open("./categorized_data/ALL_d_art_lowEst.p", "rb"))
# print(ds)
# plt.hist([d['low_rel_high'] for d in output_dicts if d['low_rel_high']==d['low_rel_high']], bins=100)
# plt.title("Distribution of Low Estimate / High Estimate")
# plt.xlabel("Low Estimate / High Estimate")
# plt.ylabel("Count")
# plt.show()


# dicts = pickle.load(open("/Users/brucewen/Desktop/honors_thesis/estimation/combined_data/categorized_data/ALL_d_art_chinese_high.p","rb"))
# print(dicts)
# n4 = [d for d in dicts if d['n']==4]
# plt.hist([d['low_rel_high'] for d in n4], bins=100)
# plt.show()

dicts = pickle.load(open("/Users/brucewen/Desktop/honors_thesis/estimation/combined_data/categorized_data/ALL_d_nyc_art_imp20.p","rb"))
high_estimates = []
for d in dicts:
    if d['n']==2:
        high_estimates.append(d['high_estimate'])
print(np.median(high_estimates))
print(np.mean(high_estimates))
print(len(high_estimates))