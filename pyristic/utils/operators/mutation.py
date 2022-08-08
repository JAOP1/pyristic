import numpy as np

__all__ = [
    'InsertionMutator','ExchangeMutator',\
    'BoundaryMutator','UniformMutator',\
    'NoneUniformMutator','BinaryMutator',\
    'NoneMutator','SigmaMutator',\
    'MultSigmaAdaptiveMutator','SingleSigmaAdaptiveMutator',\
    'SigmaEpAdaptiveMutator', 'sigma_ep_adaptive',\
    'single_sigma_adaptive', 'mult_sigma_adaptive',\
    'mutation_by_sigma', 'insertion_mutation',\
    'insertion_mutation', 'exchange_mutation',\
    'boundary_mutation', 'uniform_mutation',\
    'none_uniform_mutation', 'binary_mutation'
]


"""
---------------------------------------------------------------------------------
                                Evolution Strategy.
---------------------------------------------------------------------------------
"""
#Mutation operators.
def sigma_ep_adaptive(X: np.ndarray, alpha:float) -> np.ndarray:
    """
    ------------------------------------------------------
    Description:
        Mutation operator for Evolutionary programming.
    ------------------------------------------------------ 
    """
    return X * (1.0 + alpha * np.random.normal(0,1,size=X.shape))

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

    ------------------------------------------------------
    """
    exponent = np.random.normal(0,1) * gamma
    return np.exp(exponent) * X

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
    ------------------------------------------------------
    """
    firts_ = np.random.normal(0,1) * gamma_prime
    second_ = np.random.normal(0,1,size=X.shape) * gamma

    exponent = firts_ + second_
    return X* np.exp(exponent)

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
    normal_value = np.random.normal(0,1)
    return  X +   sigma * normal_value

"""
---------------------------------------------------------------------------------
                                Genetic Algortihms.
---------------------------------------------------------------------------------
"""

#Discrete mutation.
def insertion_mutation(X : np.ndarray, n_elements: int=1) -> np.ndarray:
    num_individuals    = len(X)
    decision_variables = len(X[0])

    X_mutated = np.ones((num_individuals,decision_variables))

    for ind in range(num_individuals):
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

def exchange_mutation(X : np.ndarray) -> np.ndarray:
    num_individuals    = len(X)
    decision_variables = len(X[0])

    for ind in range(num_individuals):
        exchange_points = np.random.choice(decision_variables,2,replace=False)
        x1 = exchange_points[0]
        x2 = exchange_points[1]

        tmp_point= X[ind][x1]
        X[ind][x1] = X[ind][x2]
        X[ind][x2] = tmp_point

    return X


#Continuos mutation.
def binary_mutation(X: np.ndarray, pm: float) -> np.ndarray:
    num_individuals = len(X)
    decision_variables = len(X[0])
    for i_individual in range(num_individuals):
        for i_variable in range(decision_variables):
            if np.random.rand() < pm:
                num = 0
                if X[i_individual][i_variable] == 0:
                    num = 1
                X[i_individual][i_variable] = num
            
    return X

def boundary_mutationArray(X: np.ndarray, lower_bound:list,\
                      upper_bound:list) -> np.ndarray:
    num_individuals = len(X)
    decision_variables = len(X[0])

    for ind in range(num_individuals):
        variable = np.random.randint(0,decision_variables)
        LB = lower_bound[variable]
        UB = upper_bound[variable]
        X[ind][variable] = LB
        if np.random.rand()>0.5:
            X[ind][variable] = UB
    return X

def boundary_mutation(X: np.ndarray, lower_bound,\
                      upper_bound) -> np.ndarray:
    num_individuals = len(X)
    decision_variables = len(X[0])
    LB,UB = lower_bound, upper_bound
    for ind in range(num_individuals):
        variable = np.random.randint(0,decision_variables)
        X[ind][variable] = LB
        if np.random.rand()>0.5:
            X[ind][variable] = UB
    return X

def uniform_mutationArray(X: np.ndarray, lower_bound: list,\
                     upper_bound: list) -> np.ndarray:
    num_individuals = len(X)
    decision_variables = len(X[0])

    for ind in range(num_individuals):
        variable = np.random.randint(0,decision_variables)
        LB = lower_bound[variable]
        UB = upper_bound[variable]    
        X[ind][variable] = np.random.uniform(LB, UB)
    return X

def uniform_mutation(X: np.ndarray, lower_bound,\
                     upper_bound) -> np.ndarray:
    num_individuals = len(X)
    decision_variables = len(X[0])

    LB,UB = lower_bound, upper_bound
    for ind in range(num_individuals):
        variable = np.random.randint(0,decision_variables)   
        X[ind][variable] = np.random.uniform(LB, UB)
    return X

def none_uniform_mutation(X:np.ndarray, sigma: float=1.0) -> np.ndarray:

    num_individuals    = len(X)
    decision_variables = len(X[0])

    for ind in range(num_individuals):
        noise = np.random.normal(0,sigma, size= decision_variables)
        X[ind] *= noise

    return X

class InsertionMutator:
    """
    Description:
      Class mutation operator based on insertion_mutation.
    Arguments:
        - n_elements:
    """
    def __init__(self, n_elements: int=1):
        self.n_elements = n_elements
        self.__doc__ = f"Insertion \n\t Arguments:\n\t\t -n_elements: {n_elements}"

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return insertion_mutation(X, self.n_elements)

class ExchangeMutator:
    """
    Description:
      Class mutation operator based on exchange_mutation.
    Arguments:
        This method doesn't need arguments.
    """
    def __init__(self):
        self.__doc__ = "Exchange"

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return exchange_mutation(X)

class BoundaryMutator:
    """
    Description:
      Class mutation operator based on boundary_mutation.
    Arguments:
        - bounds:
    """
    def __init__(self, bounds:list):
        self.Bounds = bounds
        self.typeBound =(list if type(self.Bounds[0]) == list else float)
        self.__doc__="Boundary\n\t Arguments:\n\t\t -Lower bound: {}\n\t\t -Upper bound: {}".format(self.Bounds[0],self.Bounds[1])

    def __call__(self, X:np.ndarray) -> np.ndarray:
        if self.typeBound != list:
            return boundary_mutation(X, self.Bounds[0], self.Bounds[1])

        return boundary_mutationArray(X, self.Bounds[0], self.Bounds[1])

class UniformMutator:
    """
    Description:
      Class mutation operator based on unfirom_mutation.
    Arguments:
        - bounds:
    """
    def __init__(self, bounds: list):
        self.Bounds = bounds
        self.typeBound =(list if type(self.Bounds[0]) == list else float)
        self.__doc__ = "Uniform\n\t Arguments:\n\t\t -Lower bound: {}\n\t\t -Upper bound: {}".format(self.Bounds[0],self.Bounds[1])

    def __call__(self, X:np.ndarray) -> np.ndarray:
        
        if self.typeBound != list:
            return uniform_mutation(X, self.Bounds[0], self.Bounds[1])

        return uniform_mutationArray(X, self.Bounds[0], self.Bounds[1])

class NoneUniformMutator:
    """
    Description:
      Class mutation operator based on none_uniform_mutation.
    Arguments:
        - sigma:
    """
    def __init__(self, sigma: float=1.0):
        self.sigma = sigma
        self.__doc__ = "Non Uniform\n\t Arguments:\n\t\t -Sigma: {}".format(self.sigma)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return none_uniform_mutation(X,self.sigma)

class NoneMutator:
    """
    Description:
      Class mutation operator based on none_mutation.
    Arguments:
        This methos doesn't need arguments.
    """
    def __init__(self):
        self.__doc__ = "None"

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return X

class BinaryMutator:
    """
    Description:
      Class mutation operator based on insertion_mutation.
    Arguments:
        - n_elements:
    """
    def __init__(self, pm: float = 0.2):
        self.__doc__ = f"Binary mutation \n\t Arguments:\n\t\t - probability to flip: {pm}"
        self.pm = pm
    def __call__(self, X:np.ndarray) -> np.ndarray:
        return binary_mutation(X, self.pm)

#Mutatio operator for EP.
class SigmaEpAdaptiveMutator:
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
class SigmaMutator:
    def __init__(self):
        self.__doc__ = "Sigma"

    def __call__(self, X: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
        return mutation_by_sigma(X,Sigma)

class MultSigmaAdaptiveMutator:
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

class SingleSigmaAdaptiveMutator:
    def __init__(self, decision_variables):
        self.__doc__ = "Single Sigma"
        self._length = 1
        self._tau = 1/ np.sqrt(self._length)

    @property
    def length(self):
        return self._length
    
    def __call__(self, sigma: np.ndarray) -> np.ndarray:
        return single_sigma_adaptive(sigma , self._length)