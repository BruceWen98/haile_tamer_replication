import numpy as np
from numpy import log
import matplotlib.pyplot as plt

def t(x,n):
    return ( n*(x**(n-1) - x**n) + x**n ) ** (1/n)

def tn(x,n):
    return ( n*(x**(n-1) - x**n) + x**n )

def f(n,x):
    # return np.log(x) * x**n + (x**(n-1) - x**n) * (-n*np.log(x) + n*x - n - x)
    # return tn(x,n) * np.log(t(x,n) / x) + (x**(n-1) - x**n) * (n*x-n-x)
    # return tn(x,n) * np.log(t(x,n) / x) + (x**(n-1) - x**n) * ((n-1)*np.log(x)-1)
    # return -x**(n-1)*np.log(x) + x**(n) - x**(n-1)
    # return -np.log(x) - 1 + n*np.log(t(x,n)) + 1 + np.log(t(x,n)) * (1-n)
    # return tn(x,n) * (log(t(x,n)) - 1/n*log(x) - 1/n)  + x**n * (-(n-1)/n * log(x) + 1/n)
    # return tn(x,n) * np.log(t(x,n) / x) + (x**(n-1) - x**n) * ((n-1)*(x-1)-1)
    return log(x) + 1/n*(1-1/x**n)*(-1+(n-1)*(-log(x)))
    

phis = np.linspace(1,10,100)

plt.plot(phis, f(3,phis))
plt.show()