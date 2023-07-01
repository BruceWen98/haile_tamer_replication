from KDEpy import FFTKDE    # Kernel Density Estimation
from scipy.stats import norm
import numpy as np
import math
import pickle

ds = pickle.load(open("/Users/brucewen/Desktop/honors_thesis/estimation/combined_data/categorized_data_withXi/art_imp20_nyc_high_159obs.p", "rb"))
ds = ds['data']

print(sorted([d['1'] - d['2'] for d in ds]))