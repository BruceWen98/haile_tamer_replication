import numpy as np
import scipy.integrate as integrate
from scipy.optimize import minimize

# phi function
def integrate_0_to_phi(n,phi):
    def f(s): 
        return (s**(n-2) * (1-s))
    return n * (n-1) * integrate.quad(f,0,phi)[0]

def calc_phi(n,H):
    bds = [(0,1)]
    return minimize(lambda phi: (integrate_0_to_phi(n,phi)-H)**2, x0=[0.7], method="Powell", bounds = bds).x[0]

H = np.linspace(0,1,1000)
n1_bar = 5

def calc_expression(n1,H):
    return H + n1 * calc_phi(n1+1,H)**(n1+1) - (n1+1) * calc_phi(n1,H)**n1


print([calc_expression(n1_bar,h) for h in H])