import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

output_dicts = pickle.load(open("out_dicts_ALL.p", "rb"))
print(len(output_dicts))
df = pd.DataFrame(output_dicts)

df = df.groupby(['cat0', 'cat1', 'loc']).size()
print(df.to_latex())