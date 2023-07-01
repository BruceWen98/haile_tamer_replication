import math
import numpy as np
from scipy.optimize import fsolve

def f(n,t,y):
    return t**n - n*y**((n-1)/n) + (n-1)*y
    # return t**n - n*y**(n-1) + (n-1)*y**n

ts = np.linspace(0,1,100)

# ns = [2,3,4,5,6,7,8,9,10,11,12.5,13.5,14,15,16,17,18,19,20]
ns = [2,3,4,5,6]
plots = {}

for n in ns:
    ys = []
    for t in ts:
        fun = lambda y: f(n,t,y)
        res = fsolve(fun, 0)
        ys.append(res[0])
    plots[n] = ys
    
    
import matplotlib.pyplot as plt
for n in ns:
    plt.plot(ts,plots[n])
plt.legend(ns)
plt.show()
