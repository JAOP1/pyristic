from numba import jit,prange
import numpy as np 
import math 


"""
--------------------------------------------------------------------------
                            Beale function.
--------------------------------------------------------------------------
"""
@jit(nopython=True, parallel=False)
def beale_function(X : np.ndarray) -> float:
    a = (1.5 - X[0] + X[0]*X[1])**2
    b = (2.25 - X[0] + X[0]*X[1]**2)**2
    c = (2.625 - X[0] + X[0]*X[1]**3)**2
    return a+b+c


def constraint1_beale(X: np.ndarray) -> bool:
    for i in range(len(X)):
        if -4.5>X[i] or X[i] > 4.5:
            return False 
    #If you want to see the result.
    constraint1_beale.__doc__="x1: -4.5 <= {:.2f} <= 4.5 \n x2: -4.5 <= {:.2f} <= 4.5".format(X[0],X[1])
    return True

beale_constraints = [constraint1_beale]
beale_bounds = [-4.5,4.5]

beale_ = {'function':beale_function, 'constraints': beale_constraints,\
          'bounds': beale_bounds,'decision_variables':2}

"""
--------------------------------------------------------------------------
                            Ackley function.
--------------------------------------------------------------------------
"""
@jit(nopython=True,parallel=False)
def ackley_function(X : np.ndarray) -> float:
    n = len(X)
    square_sum = (1/n)*np.sum(X * X)
    trigonometric_sum = (1/n)*np.sum(np.cos(2*np.pi*X))
 
    return -20*np.exp( -0.2 * np.sqrt(square_sum)) - np.exp(trigonometric_sum) + 20 + np.e

def constraint1_ackley(X : np.ndarray) -> bool:
    str_ = ""
    valid = True
    for i in range(len(X)):
        if -30>X[i] or X[i] > 30:
            valid = False
        str_+= "x{}: -30 <= {:.2f} <= 30 \n ".format(i+1, X[i])
    
    #Important if you want to see the result.    
    constraint1_ackley.__doc__= str_
    return valid

ackley_constraints = [constraint1_ackley]
ackley_bounds = [-30.0,30.0]

ackley_ = {'function': ackley_function, 'constraints': ackley_constraints,\
           'bounds': ackley_bounds, 'decision_variables': 10}

"""
--------------------------------------------------------------------------
                            Bukin N.6 function.
--------------------------------------------------------------------------
"""

@jit(nopython=True, parallel=True)
def bukin_function(X: np.ndarray) -> float:
    A = np.sqrt( np.abs(X[1] - 0.01* X[0]**2) ) 
    B = np.abs(X[0] + 10)
    return 100.0 * A + 0.01* B

def constraint1_bukin(X : np.ndarray) -> bool:
    str_ = ""
    str_+= "x{}: -15 <= {:.2f} <= -5 \n ".format(1, X[0])    
    str_+= "x{}: -3 <= {:.2f} <= 3 \n ".format(2, X[1])
    constraint1_bukin.__doc__= str
    if -15>X[0] or X[0] > -5:
        return False
    if -3 > X[1] or X[1] > 3:
        return False
    return True

bukin_constraints = [constraint1_bukin]
bukin_bounds = [[-15,-3],[-5,3]] # [[Lower bounds] , [Upper bounds]]

bukin_ = {'function': bukin_function, 'constraints': bukin_constraints,\
           'bounds': bukin_bounds, 'decision_variables': 2}

"""
--------------------------------------------------------------------------
                            Himmelblau's function.
--------------------------------------------------------------------------
"""


@jit(nopython=True, parallel=True)
def Himmelblau_function(X: np.ndarray) -> float:
    A = (X[0]**2 + X[1] - 11)**2
    B = (X[0] + X[1]**2 - 7)**2
    return A+B

def constraint1_Himmelblau(X : np.ndarray) -> bool:
    str_ = ""
    valid = True
    for i in range(len(X)):
        if -5>X[i] or X[i] > 5:
            valid = False
        str_+= "x{}: -5 <= {:.2f} <= 5 \n ".format(i+1, X[i])
    
    #Important if you want to see the result.    
    constraint1_ackley.__doc__= str_
    return valid

Himmelblau_constraints = [constraint1_Himmelblau]
Himmelblau_bounds = [-5,5]

Himmelblau_ = {'function': Himmelblau_function, 'constraints': Himmelblau_constraints,\
           'bounds': Himmelblau_bounds, 'decision_variables': 2}