## Taken from: https://towardsdatascience.com/integrals-are-fun-illustrated-riemann-stieltjes-integral-b71a6e003072

def derivative(f, a, h=0.01):
    '''Approximates the derivative of the function f in a
    
    :param function f: function to differentiate
    :param float a: the point of differentiation
    :param float h: step size
    :return float: the derivative of f in a
    '''
    return (f(a + h) - f(a - h))/(2*h)
  
def stieltjes_integral(f, g, a, b, n, h):
    '''Calculates the Riemann-Stieltjes integral based on the composite trapezoidal rule
    relying on the Riemann Sums.
    
    :param function f: integrand function
    :param function g: integrator function 
    :param int a: lower bound of the integral
    :param int b: upper bound of theintergal
    :param int n: number of trapezoids of equal width
    :return float: the integral of the function f between a and b
    '''
    eps = 1e-9
    h = (b - a)/(n + eps)  # width of the rectangle
    dg = lambda x: derivative(g, x, h=h)  # derivative of the integrator function
    result = 0.5*f(a)*dg(a) + sum([f(a + i*h)*dg(a + i*h) for i in range(1, n)]) + 0.5*f(b)*dg(b)
    result *= h
    return result