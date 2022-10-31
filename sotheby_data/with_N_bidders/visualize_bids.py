import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

with open('output_dicts_ALL_{}.pickle'.format("withN"), 'rb') as handle:
    output_dicts_ALL = pickle.load(handle)
    
print(output_dicts_ALL)

ns = np.unique([x["n"] for x in output_dicts_ALL])
for n in ns[-1:]:
    top1bid = []
    top2bid = []
    for output_dict in output_dicts_ALL:
        if output_dict["n"] == n:
            top1bid.append(output_dict["1"])
            top2bid.append(output_dict["2"])
    print(top1bid,top2bid)
        
plt.plot(np.sort(top1bid), np.linspace(0, 1, len(top1bid), endpoint=False), label="Top1 Bid")
plt.plot(np.sort(top2bid), np.linspace(0, 1, len(top2bid), endpoint=False), label="Top2 Bid")
plt.legend()
plt.show()