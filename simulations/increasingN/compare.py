import math
import numpy as np
from scipy.optimize import fsolve

def f(n,x):
    return np.log(x) + 1/n * (1-1/(x**n)) * (-1-(n-1)*np.log(x))
    # return t**n - n*y**(n-1) + (n-1)*y**n

ts = np.linspace(1,10,1000)

# ns = [2,3,4,5,6,7,8,9,10,11,12.5,13.5,14,15,16,17,18,19,20]
ns = [2,3,4,5,6]
plots = {}

for n in ns:
    ys = f(n,ts)
    plots[n] = ys
    
    
import matplotlib.pyplot as plt
for n in ns:
    plt.plot(ts,plots[n])
plt.legend(ns)
plt.show()