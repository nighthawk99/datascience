# -*- coding: utf-8 -*-


import mathplotlib.pyplot as plt



def sum_of_squares(v):
    return sum(v_i ** 2 for v_i in v)

def difference_quotient(f,x,h):
    return(f(x+h)-f(x))/h
    
def square(x):
    return x*x

def square_derivative(x):
    return 2*x

derivative_estimate = partial(difference_quotient,square,h=0.0001)


x=range(-10,10)
plt.title("Actual Derivative vs.. Estimates")
plt.plot(x, map(derivative_estimate,x),'b+',label='Estimate')
plt.legend(loc=9)
plt.show


    
