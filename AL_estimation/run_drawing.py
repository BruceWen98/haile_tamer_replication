##### This script runs the drawing function for any of:
# i) draw_AL_profits_specific.py
# ii) draw_non_equilibrium_profits.py
# iii) draw_profit_diff.py

import draw_AL_profits_specific as DAPS
import draw_non_equilibrium_profits as DNEP
import draw_profit_diff as DPD

import sys
import os
from pathlib import Path

f = sys.argv[1]
option = sys.argv[2]

print("Working on {} now...".format(f))
INPATH = str(f)
OUTPATH = "/Users/brucewen/Desktop/honors_thesis/estimation/combined_data/categorized_data_results_withXi/" + INPATH.split("/")[-1].split(".")[0] + "/"
SELECT_RESERVE_PATH = "/Users/brucewen/Desktop/honors_thesis/selecting reserve/code/computed_bounds/"
if Path(OUTPATH).is_dir()==False:
    os.mkdir(OUTPATH)

print(OUTPATH)

if option == "DAPS":
    DAPS.run_AL_profits_specificN_Data(INPATH, OUTPATH, n=3, UB_V=5, num_points=1000)
elif option == "DNEP":
    DNEP.run_AL_profits_specificN_Data(INPATH, OUTPATH, n=3, UB_V=10, num_points=500)
elif option == "DPD":
    DPD.draw_profit_diff_specificN(INPATH, OUTPATH, n=3, ub_v=5, num_points=300)