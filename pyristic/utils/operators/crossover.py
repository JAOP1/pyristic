import numpy as np 
from numba import jit,prange
from numba.typed import List

__all__ = ['intermediate_crossover','n_point_crossover',\
           'uniform_crossover','permutation_order_crossover','simulated_binary_crossover',\
            'discrete_cross', 'intermediate_cross', 'n_point_cross', 'uniform_cross', 'permutation_order_cross',\
            'simulated_binary_cross']

"""
---------------------------------------------------------------------------------
                                Evolution Strategy.
---------------------------------------------------------------------------------
"""

#Crossover operators.
@jit(nopython=False,parallel=True)
def discrete_cross(X : np.ndarray, parent_ind1: np.ndarray ,\
                       parent_ind2: np.ndarray) -> np.ndarray:
    """
    ------------------------------------------------------
    Description:
        Function which create new elements combining two elements.
        X_{i,s} if v == 1 
        X_{i,t} if v == 0
    Arguments:
        -X: Matrix with size m x n, where m is the current size of
        population and n is the problem variables. Every row is an element.
        -parent_ind1: the chosen parents, where every position is the index in X.
        -parent_ind2: the chosen parents, where every position is the index in X.
    ------------------------------------------------------
    """

    rows,cols = X.shape
    rows_new = len(parent_ind1)
    A = np.zeros((rows_new,cols))
    B = np.zeros((rows_new,cols))

    for r in prange(rows_new):
        for c in prange(cols):
            if np.random.randint(2):
                A[r,c] = 1
            else:
                B[r,c] = 1

    return X[parent_ind1] * A + X[parent_ind2] * B

@jit(nopython=False,parallel=True)
def intermediate_cross(X:np.ndarray, parent_ind1: np.ndarray,\
                       parent_ind2: np.ndarray, alpha: float=0.5) -> np.ndarray:
    """
    ------------------------------------------------------
    Description:
        Function which create new elements combining  the average of two elements.
    Arguments:
        -X: Matrix with size m x n, where m is the current size of
        population and n is the problem variables. Every row is an element.
        -parent_ind1: the chosen parents, where every position is the index in X.
        -parent_ind2: the chosen parents, where every position is the index in X.
    ------------------------------------------------------
    """
    return alpha*X[parent_ind1]+ (1-alpha) * X[parent_ind2]

"""
---------------------------------------------------------------------------------
                                Genetic Algorithm.
---------------------------------------------------------------------------------
"""

@jit(nopython=True,parallel=True)
def n_point_cross(X : np.ndarray , parent_ind1 : np.ndarray,\
                  parent_ind2 : np.ndarray, n_cross: int = 1) -> np.ndarray:
    """
    ------------------------------------------------------
    Description:
        Function which create new elements combining two individuals.
    Arguments:
        -X: Matrix with size m x n, where m is the current size of
        population and n is the problem variables. Every row is an element.
        -parent_ind1: the chosen parents, where every position is the index in X.
        -parent_ind2: the chosen parents, where every position is the index in X.
        -n_cross: integer value which says you the number of cuts (by default is 1).

        Note: 
            parent_ind1 and parent_ind2 should be m elements.
    Return:
        Matriz with size 2m x n.
    ------------------------------------------------------
    """

    num_individuals = len(parent_ind1)
    decision_variables = len(X[0])    
    new_population = np.ones((num_individuals*2,decision_variables), dtype=np.float64)

    for i in prange(0,num_individuals*2):
        if i%2 == 0:
            new_population[i] =  X[parent_ind1[(i//2)]]
        else:
            new_population[i] = X[parent_ind2[(i//2)]]

    for row in prange(num_individuals):
        cross_points  = np.random.choice(decision_variables-1,n_cross,replace=False)
        cross_points += 1
        cross_points.sort()
        for i in range(0,n_cross,2):
            start = cross_points[i]
            final = decision_variables 
            if i+1 < n_cross:
                final = cross_points[i+1]
            new_population[2*row][start:final]    = X[parent_ind2[row]][start:final]
            new_population[2*row +1][start:final] = X[parent_ind1[row]][start:final]

    return new_population

@jit(nopython=True, parallel=True)
def uniform_cross(X: np.ndarray, parent_ind1: np.ndarray,\
                  parent_ind2 : np.ndarray, flip_prob : int=0.5) -> np.ndarray:
    """
    ------------------------------------------------------
    Description:
        Function which create new elements combining two individuals.
    Arguments:
        -X: Matrix with size m x n, where m is the current size of
        population and n is the problem variables. Every row is an element.
        -parent_ind1: the chosen parents, where every position is the index in X.
        -parent_ind2: the chosen parents, where every position is the index in X.
        -flip_prob: float value which says the likely to change a position (by default is 0.5).
        Note: 
            parent_ind1 and parent_ind2 should be m elements.
    Return:
        Matriz with size 2m x n.
    ------------------------------------------------------
    """

    num_individuals = len(parent_ind1)
    decision_variables = len(X[0])
    new_population = np.ones((num_individuals*2,decision_variables),dtype=np.float64)

    for i in prange(0,num_individuals*2):
        if i%2 == 0:
            new_population[i] =  X[parent_ind1[(i//2)]]
        else:
            new_population[i] = X[parent_ind2[(i//2)]]

    for i in prange(num_individuals):
        for x in range(decision_variables):
            if np.random.rand() <= flip_prob:
                new_population[2*i][x] = X[parent_ind2[i]][x]
                new_population[2*i+1][x] = X[parent_ind1[i]][x]

    return new_population

@jit(nopython=False, parallel=True)
def create_child(parent1: np.ndarray , parent2: np.ndarray) -> np.ndarray:

    """
        ¡¡¡FUNCTION OF permutation_order_cross!!!
    """
    decision_variables = len(parent1)
    interval = np.random.choice(decision_variables+1,2,replace=False)
    interval.sort()

    individual = np.full(decision_variables,np.inf)
    segment = parent1[interval[0]:interval[1]]
    individual[interval[0]:interval[1]] = segment

    remainder = []
    for x in parent2:
        if not x in segment:
            remainder.append(x)

    individual[individual == np.inf] = remainder
    return individual

@jit(nopython=False, parallel=True)
def permutation_order_cross(X : np.ndarray , parent_ind1:np.ndarray,\
                            parent_ind2:np.ndarray) -> np.ndarray:
    """
    ------------------------------------------------------
    Description:
        Function which create new elements combining two individuals.
    Arguments:
        -X: Matrix with size m x n, where m is the current size of
        population and n is the problem variables. Every row is an element.
        -parent_ind1: the chosen parents, where every position is the index in X.
        -parent_ind2: the chosen parents, where every position is the index in X.
        Note: 
            parent_ind1 and parent_ind2 should be m elements.
    Return:
        Matriz with size 2m x n.
    ------------------------------------------------------
    """

    num_individuals = len(parent_ind1)
    decision_variables = len(X[0])
    new_population = np.ones((num_individuals*2,decision_variables))

    for i in prange(num_individuals):
        new_population[2*i]  = create_child(X[parent_ind1[i]], X[parent_ind2[i]])
        new_population[2*i+1] = create_child(X[parent_ind2[i]], X[parent_ind1[i]])
    
    return new_population

@jit(nopython=True, parallel=True)
def simulated_binary_cross(X: np.ndarray, parent_ind1: np.ndarray,\
                           parent_ind2: np.ndarray,\
                           nc: int=1) -> np.ndarray:

    num_individuals = len(parent_ind1)
    decision_variables = len(X[0])                       

    u = np.random.rand()
    B = -1
    exponent = 1/(nc+1)
    if u <= 0.5:
        B = np.power((2*u), exponent)
    else:
        B = np.power(1/(2 * (1-u)), exponent)
    X_1 =np.ones((num_individuals,decision_variables), dtype=np.float64)
    X_2 = np.ones((num_individuals,decision_variables), dtype=np.float64)
    for  i in prange(0, num_individuals):
        X_1[i] = X[parent_ind1[i]]
        X_2[i] = X[parent_ind2[i]]

    A = X_1 + X_2
    B = np.absolute(X_2 - X_1) * B

    return np.concatenate((0.5 * (A - B) , 0.5 * (A + B)) , axis = 0)


class intermediate_crossover:
    def __init__(self, alpha: float=0.5):
        self.alpha = alpha
        self.__doc__ = "Intermediate\n\tArguments:\n\t\t-Alpha:{}".format(self.alpha)
    
    def __call__(self,  population:np.ndarray,\
                        parent_ind1: np.ndarray,\
                        parent_ind2: np.ndarray) -> np.ndarray:
        assert len(parent_ind1) == len(parent_ind2)
        return intermediate_cross(population,parent_ind1,\
                                parent_ind2, self.alpha)

class n_point_crossover:
    def __init__(self, n_cross: int = 1):
        self.n_cross = n_cross
        self.__doc__ = "n point\n\tArguments:\n\t\t-n_cross: {}".format(self.n_cross)

    def __call__(self,  population:np.ndarray,\
                    parent_ind1: np.ndarray,\
                    parent_ind2: np.ndarray) -> np.ndarray:
        assert len(parent_ind1) == len(parent_ind2)
        
        return n_point_cross(population, List(parent_ind1), List(parent_ind2), self.n_cross)

class uniform_crossover:
    def __init__(self,flip_prob : int=0.5):
        self.prob = flip_prob
        self.__doc__ = "Uniform\n\tArguments:\n\t\t-prob: {}".format(self.prob)

    def __call__(self,  population:np.ndarray,\
                    parent_ind1: np.ndarray,\
                    parent_ind2: np.ndarray) -> np.ndarray:
        assert len(parent_ind1) == len(parent_ind2)    
        return uniform_cross(population, List(parent_ind1), List(parent_ind2), self.prob)

class permutation_order_crossover:
    def __init__(self):
        self.__doc__ = "Permutation order"

    def __call__(self,  population:np.ndarray,\
                    parent_ind1: np.ndarray,\
                    parent_ind2: np.ndarray) -> np.ndarray:
        assert len(parent_ind1) == len(parent_ind2)
        return permutation_order_cross(population, parent_ind1, parent_ind2)

class simulated_binary_crossover:
    def __init__(self,nc: int=1):
        self.nc = nc
        self.__doc__ = "Simulated binary\n\tArguments:\n\t\t-nc: {}".format(self.nc)

    def __call__(self,  population: np.ndarray,\
                parent_ind1: np.ndarray,\
                parent_ind2: np.ndarray) -> np.ndarray:
        assert len(parent_ind1) == len(parent_ind2)

        return simulated_binary_cross(List(population), List(parent_ind1),List(parent_ind2), self.nc)

class none_cross_crossover:
    def __init__(self):
        self.__doc__ = "None"

    def __call__(self, population : np.ndarray,\
                       parent_ind1: np.ndarray,\
                       parent_ind2: np.ndarray) -> np.ndarray:
        return population

#Crossover operator for ES.
class discrete_crossover:
    def __init__(self):
        self.__doc__ = "Discrete"
    
    def __call__(self, population : np.ndarray,\
                       parent_ind1: np.ndarray,\
                       parent_ind2: np.ndarray) -> np.ndarray:
        assert len(parent_ind1) == len(parent_ind2)
        # p1 = List(parent_ind1)
        # p2 = List(parent_ind2)
        return discrete_cross(population, parent_ind1, parent_ind2)

    