import numpy as np
import scipy.integrate as integrate
from scipy.optimize import minimize
import math

## 1) Numerical Integration of the pdf of the i:n order statistic.
def integrate_0_to_phi(i,n,phi):
    def f(s):
        return (s**(i-1) * (1-s)**(n-i))
    return integrate.quad(f,0,phi)

def calc_H(i,n,phi):
    integral = integrate_0_to_phi(i,n,phi)[0]
    scale = math.factorial(n) / (math.factorial(i-1) * math.factorial(n-i))
    return integral * scale

def calc_phi(i,n,H):
    bds = [(0,1)]
    return minimize(lambda phi: (calc_H(i,n,phi)-H)**2, x0=[0.7], method="Powell", bounds = bds).x[0]