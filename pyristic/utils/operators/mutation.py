import numpy as np 
from numba import jit,prange
from numba.typed import List
import numba

__all__ = ['insertion_mutator','exchange_mutator','boundary_mutator','uniform_mutator',\
           'non_uniform_mutator','none_mutator','sigma_mutator','mult_sigma_adaptive_mutator',\
           'single_sigma_adaptive_mutator','sigma_ep_adaptive_mutator', 'sigma_ep_adaptive',\
            'single_sigma_adaptive', 'mult_sigma_adaptive', 'mutation_by_sigma', 'insertion_mutation',\
            'insertion_mutation', 'exchange_mutation', 'boundary_mutation', 'uniform_mutation',\
            'non_uniform_mutation']


"""
---------------------------------------------------------------------------------
                                Evolution Strategy.
---------------------------------------------------------------------------------
"""
#Mutation operators.
@jit(nopython=True, parallel=True)
def sigma_ep_adaptive(X: np.ndarray, alpha:float) -> np.ndarray:
    return X * (1.0 + alpha * np.random.normal(0,1,size=X.shape))

@jit(nopython=True,parallel=True)
def single_sigma_adaptive(X: np.ndarray,gamma:float) -> np.ndarray:
    """
    ------------------------------------------------------
    Description:
        Every individual has a sigma parameter which helps you to 
        update in a fixed portion the individual.
        
        Note:
            This function is for Evolution Strategy search. 
    Arguments: 
        - X: Numpy array where every N-th row is an individual and 
          M columns that are the decision variables.
        - gamma: Float value, additional parameter. 

        About:
            If you want more information, check:
            ¡INCLUIR ARTICULO SOBRE ESTO!
    ------------------------------------------------------ 
    """
    exponent = np.random.normal(0,1) * gamma
    return np.exp(exponent) * X

@jit(nopython=True, parallel=True)
def mult_sigma_adaptive(X: np.ndarray, gamma: float, gamma_prime: float) -> np.ndarray:
    """
    ------------------------------------------------------
    Description:
        Every individual has a sigma for every decision variable which helps you to 
        update in a fixed portion the individual.

        Note: 
            This function is for Evolution Strategy search.
    Arguments: 
        - X: Numpy array where every N-th row is an individual and 
          M columns that are the decision variables.
        - gamma: Float value, additional parameter. 
        - gamma_prime: Float value, additional parameter.

    About: 
        If you want more information, check:
        ¡INCLUIR ARTICULO SOBRE ESTO!
    ------------------------------------------------------ 
    """
    firts_ = np.random.normal(0,1) * gamma_prime 
    second_ = np.random.normal(0,1,size=X.shape) * gamma

    exponent = firts_ + second_
    return X* np.exp(exponent)   


@jit(nopython=True,parallel=False)
def mutation_by_sigma(X: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """
    ------------------------------------------------------
    Description:
        Update function in the decision variables of every individual.

        Note: 
            This function is for Evolution Strategy search.
    Arguments: 
        - X: Numpy Matrix, where every N-th row is an individual and 
          M columns that are the decision variables.
        - sigma: Numpy Array with N rows which means every row is a individual, M columns differ in
        what kind of sigma has chosen (single or multiple).
    ------------------------------------------------------ 
    """
    NormalValue = np.random.normal(0,1)
    return  X +   sigma * NormalValue

"""
---------------------------------------------------------------------------------
                                Genetic Algortihms.
---------------------------------------------------------------------------------
"""

#Discrete mutation.
@jit(nopython=True, parallel=True)
def insertion_mutation(X : np.ndarray, n_elements: int=1) -> np.ndarray:
    num_individuals    = len(X)
    decision_variables = len(X[0])

    X_mutated = np.ones((num_individuals,decision_variables))

    for ind in prange(num_individuals):
        individual = np.full(decision_variables,np.inf)
        indices_elements = np.random.choice(decision_variables,n_elements, replace=False)
        elements = X[ind][indices_elements]
        indices_position = np.random.choice(decision_variables,n_elements, replace= False)
        individual[indices_position] = elements

        indices_elements.sort()
        remaining = []
        start_ = 0
        for e in range(decision_variables):
            if start_ < len(indices_elements) and e == indices_elements[start_]:
                start_+=1
                continue
            remaining.append(X[ind][e])

        individual[individual == np.inf] = remaining
        X_mutated[ind] = individual

    return X_mutated

@jit(nopython=True, parallel=True)
def exchange_mutation(X : np.ndarray) -> np.ndarray:
    num_individuals    = len(X)
    decision_variables = len(X[0])

    for ind in prange(num_individuals):
        exchange_points = np.random.choice(decision_variables,2,replace=False)
        x1 = exchange_points[0]
        x2 = exchange_points[1]

        tmp_point= X[ind][x1]
        X[ind][x1] = X[ind][x2]
        X[ind][x2] = tmp_point

    return X

#Continuos mutation.
@jit(nopython=True, parallel=True)
def boundary_mutationArray(X: np.ndarray, lower_bound:list,\
                      upper_bound:list) -> np.ndarray:
    num_individuals = len(X)
    decision_variables = len(X[0])

    for ind in prange(num_individuals):
        variable = np.random.randint(0,decision_variables)
        LB = lower_bound[variable]
        UB = upper_bound[variable]
        X[ind][variable] = LB
        if np.random.rand()>0.5:
            X[ind][variable] = UB
    return X

@jit(nopython=True, parallel=True)
def boundary_mutation(X: np.ndarray, lower_bound,\
                      upper_bound) -> np.ndarray:
    num_individuals = len(X)
    decision_variables = len(X[0])
    LB,UB = lower_bound, upper_bound
    for ind in prange(num_individuals):
        variable = np.random.randint(0,decision_variables)
        X[ind][variable] = LB
        if np.random.rand()>0.5:
            X[ind][variable] = UB
    return X

@jit(nopython=True, parallel=True)
def uniform_mutationArray(X: np.ndarray, lower_bound: list,\
                     upper_bound: list) -> np.ndarray:
    num_individuals = len(X)
    decision_variables = len(X[0])

    for ind in prange(num_individuals):
        variable = np.random.randint(0,decision_variables)
        LB = lower_bound[variable]
        UB = upper_bound[variable]    
        X[ind][variable] = np.random.uniform(LB, UB)
    return X

@jit(nopython=True, parallel=True)
def uniform_mutation(X: np.ndarray, lower_bound,\
                     upper_bound) -> np.ndarray:
    num_individuals = len(X)
    decision_variables = len(X[0])

    LB,UB = lower_bound, upper_bound
    for ind in prange(num_individuals):
        variable = np.random.randint(0,decision_variables)   
        X[ind][variable] = np.random.uniform(LB, UB)
    return X

@jit(nopython=False, parallel=True)
def non_uniform_mutation(X:np.ndarray, sigma: float=1.0) -> np.ndarray:

    num_individuals    = len(X)
    decision_variables = len(X[0])

    for ind in prange(num_individuals):
        noise = np.random.normal(0,sigma, size= decision_variables)
        X[ind] *= noise

    return X

class insertion_mutator:
    def __init__(self, n_elements: int=1):
        self.n_elements = n_elements
        self.__doc__ = "Insertion \n\t Arguments:\n\t\t -n_elements: {}".format(n_elements)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return insertion_mutation(X, self.n_elements)

class exchange_mutator:
    def __init__(self):
        self.__doc__ = "Exchange"

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return exchange_mutation(X)

class boundary_mutator:
    def __init__(self, bounds:list):
        self.Bounds = bounds
        self.typeBound =(list if type(self.Bounds[0]) == list else float)
        self.__doc__="Boundary\n\t Arguments:\n\t\t -Lower bound: {}\n\t\t -Upper bound: {}".format(self.Bounds[0],self.Bounds[1])

    def __call__(self, X:np.ndarray) -> np.ndarray:
        if self.typeBound != list:
            return boundary_mutation(List(X), self.Bounds[0], self.Bounds[1])

        return boundary_mutationArray(List(X), List(self.Bounds[0]), List(self.Bounds[1]))

class uniform_mutator:
    def __init__(self, bounds: list):
        self.Bounds = bounds
        self.typeBound =(list if type(self.Bounds[0]) == list else float)
        self.__doc__ = "Uniform\n\t Arguments:\n\t\t -Lower bound: {}\n\t\t -Upper bound: {}".format(self.Bounds[0],self.Bounds[1])

    def __call__(self, X:np.ndarray) -> np.ndarray:
        
        if self.typeBound != list:
            return uniform_mutation(List(X), self.Bounds[0], self.Bounds[1])

        return uniform_mutationArray(List(X), List(self.Bounds[0]), List(self.Bounds[1]))

class non_uniform_mutator:
    def __init__(self, sigma: float=1.0):
        self.sigma = sigma
        self.__doc__ = "Non Uniform\n\t Arguments:\n\t\t -Sigma: {}".format(self.sigma)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return non_uniform_mutation(X,self.sigma)

class none_mutator:
    def __init__(self):
        self.__doc__ = "None"

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return X

#Mutatio operator for EP.
class sigma_ep_adaptive_mutator:
    def __init__(self,decision_variables:int ,alpha: float):
        self._alpha = alpha
        self._length = decision_variables
        self.__doc__ = "Sigma EP.\n\t\t-Alpha: {}".format(self._alpha)

    @property
    def length(self):
        return self._length

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return sigma_ep_adaptive(X,self._alpha)

#Mutation operator for ES.
class sigma_mutator:
    def __init__(self):
        self.__doc__ = "Sigma"

    def __call__(self, X: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
        return mutation_by_sigma(X,Sigma)

class mult_sigma_adaptive_mutator:
    def __init__(self, decision_variables):
        self.__doc__ = "Sigma mult"
        self._length = decision_variables
        self._gamma_prime = 1/np.sqrt(2*self._length)
        self._gamma = 1/ np.sqrt( 2 * np.sqrt(self._length))

    @property
    def length(self):
        return self._length
    
    def __call__(self, sigma: np.ndarray) -> np.ndarray:
        return mult_sigma_adaptive(sigma,self._gamma, self._gamma_prime)

class single_sigma_adaptive_mutator:
    def __init__(self, decision_variables):
        self.__doc__ = "Single Sigma"
        self._length = 1
        self._tau = 1/ np.sqrt(self._length)

    @property
    def length(self):
        return self._length
    
    def __call__(self, sigma: np.ndarray) -> np.ndarray:
        return single_sigma_adaptive(sigma , self._length)