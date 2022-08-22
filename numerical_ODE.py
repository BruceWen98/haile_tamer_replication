## This is code to implicitly solve for ODE numerically.
import numpy as np
import scipy.integrate as integrate
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


## 2) Derivative of H (analytical). This is by fundamental theorem of calculus.
def calc_H_deriv(i,n,phi):
    left = math.factorial(n) / (math.factorial(i-1) * math.factorial(n-i))
    right = (phi**(i-1) * (1-phi)**(n-i))
    return left * right


## 3) Solve for phi given (H; i,n) using Newton's method.
# Note that the initial guess phi is in [0,1], so we will pick something in this range randomly.
# There are 3 stopping conditions (whichever is reached first) in Newton's method:
    # 1) Residual Size, calc_H(.) is close to H
    # 2) Increment Size, |phi_new - phi_old| < \epsilon
    # 3) Iteration Limit, not more than 1000.
# At the end of the algorithm, we need to check that phi is in [0,1]. If not, restart with different initial guess.
def newton_method(i,n,H, max_iters=1000):
    phi_old = np.random.uniform(0, 1, 1)[0]  # initialize phi
    phi_new = -1 # initialize phi_new as -1, out of the distribution support.
    iters = 0

    phi_result = -1 # initialize phi_result as -1, out of the distribution support.
    while iters<max_iters:
        # if calc_h(i,n,phi_old) is sufficiently close to H
        if abs( calc_H(i,n,phi_old) - H )  < 1e-20:
            phi_result =  phi_old
            break

        # else, Newton's Method until we get close enough.
        phi_new = phi_old - (calc_H(i,n,phi_old)-H) / calc_H_deriv(i,n,phi_old)
        if abs( phi_new-phi_old ) < 1e-20:
            phi_result =  phi_new
            break

        phi_old = phi_new
        iters+=1
    
    # Recursion to make sure phi is in [0,1]. Change initial guess if fail.
    if phi_new < 0 or phi_new > 1:
        return newton_method(i,n,H,max_iters)
    else:
        return phi_result

# Account for possible Errors. Reinitialize and try again.
def newton_method_errorProof(i,n,H, max_iters=200):
    try:
        try1 = newton_method(i,n,H,max_iters)
        if try1 < 0 or try1 > 1:
            return newton_method_errorProof(i,n,H,max_iters)
        else:
            return try1
    except Exception as e:
        print(e)
        return newton_method(i,n,H,max_iters)


# testing = calc_H(10,15,0.6045)
# # print(testing)

# for i in range(1000):
#     print(newton_method_errorProof(10,15,testing))

 

