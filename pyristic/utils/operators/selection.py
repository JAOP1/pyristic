import numpy as np 
from numba import jit,prange
from copy import deepcopy

__all__ = ['proporcional_sampler', 'roulette_sampler', 'stochastic_universal_sampler',\
           'deterministic_sampler', 'tournament_sampler', 'merge_selector', 'replacement_selector',\
            'proporcional_sampling', 'roulette_sampling', 'stochastic_universal_sampling',\
            'deterministic_sampling', 'tournament_sampling']

@jit(nopython=True, parallel=True)
def get_expected_values(aptitude: np.array) -> np.array:
    averageAptitude = np.sum(aptitude)
    N = len(aptitude)
    return aptitude/averageAptitude * N 

"""
---------------------------------------------------------------------------------
                            |   Genetic Algorithms.   |
---------------------------------------------------------------------------------
"""


"""
---------------------------------------------------------------------------------
                                Parent selection.
---------------------------------------------------------------------------------
"""
@jit(nopython=True,parallel=False)
def proporcional_sampling(expectedVals: np.ndarray) -> np.ndarray:
    return np.arange(len(expectedVals))

@jit(nopython=True,parallel=True)
def roulette_sampling(expectedVals : np.ndarray) -> np.ndarray:
    """
    ------------------------------------------------------
    Description:
        selection algorithm O(nlogn) for genetic algorithm. 
    Arguments:
        probabilities: 1-D numpy array which is the probability of every 
        individual based on the aptitude.
    About:
        If you want more information, check:
        
    ------------------------------------------------------
    """
    expectedVals = get_expected_values(expectedVals)
    N = len(expectedVals)
    # expectedVals_ = np.copy(expectedVals)

    expectedCumulative = np.cumsum(expectedVals)
    r = np.random.uniform(0.0,expectedCumulative[-1], N)
    return np.searchsorted(expectedCumulative, r) #sample
    
@jit(nopython=False, parallel=False)   
def stochastic_universal_sampling(expectedVals : np.ndarray) -> np.ndarray:
    """
    ------------------------------------------------------ 
    Arguments:
        probabilities: 1-D numpy array which is the probability of every 
        individual based on the aptitude.
    About:
        If you want more information, check:
        
    ------------------------------------------------------
    """
    expectedVals = get_expected_values(expectedVals)

    N = len(expectedVals)
    # expectedVals_ =np.copy(expectedVals)

    r = np.random.uniform(0,1)
    currentSum = 0
    sample = []
    for i in range(N):
        currentSum += expectedVals[i]
        while r < currentSum:
            sample.append(i)
            r += 1
    return np.array(sample)

def deterministic_sampling(expectedVals : np.ndarray) -> np.ndarray:
    expectedVals = get_expected_values(expectedVals)

    N = len(expectedVals)

    integer_part = np.array(expectedVals, dtype=int)
    indices = np.arange(N)
    sample = np.repeat(indices,integer_part) 
    
    if(len(sample) < N):
        float_part = expectedVals - integer_part
        ind = np.argpartition(float_part,len(sample) - N)[len(sample) - N:]
        sample = np.concatenate((sample, indices[ind]))
    
    return sample[:N]

def tournament_sampling( expectedVals : np.ndarray, chunks : int=2 , prob: float=1.0) -> np.ndarray:
    """
    ------------------------------------------------------
    Description:
        selection algorithm O(NP) for genetic algorithm where N is the population
        size and P is the number of chunks. 
    Arguments:
        probabilities: 1-D numpy array which is the probability of every 
        individual based on the aptitude.
        chunks: Integer value which is the number of chunks (by default is 2).
        prob: Float value which means if choose the max or min value in every chunk
        (by default is 1). 
        About:
            If you want more information, check:
            
    ------------------------------------------------------
    """ 
    expectedVals = get_expected_values(expectedVals)
    N=len(expectedVals)
    indices = np.arange(N)
    chunks_ = np.ceil(N/chunks)

    sample = []
    ind = -1
    for tournament in range(chunks):
        groups = np.array_split(np.random.permutation(indices), chunks_)
        for i in prange(len(groups)):
            if prob >= np.random.rand():
                ind = np.argmax(expectedVals[groups[i]])
            else:
                ind = np.argmin(expectedVals[groups[i]])
            sample.append(groups[i][ind])

    return np.array(sample)
    

"""
---------------------------------------------------------------------------------
                                Survivor selection.
---------------------------------------------------------------------------------
"""
def get_lowest_indices(value: np.ndarray, m: int) -> np.ndarray:
    """
    ------------------------------------------------------
    Description:
        Return m indices with the lowest fitness.
    Arguments:
        value: float array.
        m: int element.
    ------------------------------------------------------
    """
    pseudo_sorted_array = np.argpartition(value, m)    
    return pseudo_sorted_array[:m]



"""
---------------------------------------------------------------------------------
                                Classes
---------------------------------------------------------------------------------
"""

"""
----------------------------
    Parent Selection.
----------------------------
"""


class proporcional_sampler:
    def __init__(self, transform=None):
        self.__doc__ = "Proporcional sampling"

    def __call__(self,population_f: np.ndarray) -> np.ndarray:
        return proporcional_sampling(population_f)

class roulette_sampler:
    def __init__(self, transform=None):
        self.__doc__ = "Roulette sampling"
        self.transform = transform

    def __call__(self,population_f: np.ndarray) -> np.ndarray:
        vals = np.copy(population_f)
        if self.transform != None:
            vals = self.transform(vals)
        
        return roulette_sampling(vals)

class stochastic_universal_sampler:
    def __init__(self, transform=None):
        self.__doc__   = "Stochastic universal sampling"
        self.transform = transform

    def __call__(self,population_f: np.ndarray) -> np.ndarray:
        vals = np.copy(population_f)
        if self.transform != None:
            vals = self.transform(vals)
        
        return stochastic_universal_sampling(vals)

class deterministic_sampler:
    def __init__(self, transform=None):
        self.__doc__   = "Deterministic sampling"
        self.transform = transform

    def __call__(self,population_f: np.ndarray) -> np.ndarray:
        vals = np.copy(population_f)
        if self.transform != None:
            vals = self.transform(vals)
        
        return deterministic_sampling(vals)

class tournament_sampler:
    def __init__(self, transform=None, chunks_ : int=2 , prob_: float=1.0):  
        self.transform = transform
        self.chunks = chunks_
        self.prob = prob_
        self.__doc__ = "Tournament sampling\n\t Arguments:\n\t\t-Chunks: {}\n\t\t-prob: {}".format(self.chunks,self.prob)

    def __call__(self,population_f: np.ndarray) -> np.ndarray:
        vals = np.copy(population_f)
        if self.transform != None:
            vals = self.transform(vals)
        
        return tournament_sampling( vals,\
                                    self.chunks,\
                                    self.prob)

"""
----------------------------
    Survivor Selection.
----------------------------
"""

class merge_selector:

    def __init__(self):
        self.__doc__ = "Merge population"
    
    def __call__(self, parent_f: np.ndarray,\
                       offspring_f: np.ndarray,\
                       features: dict) -> dict:
        """
        ------------------------------------------------------
        Description:
            survivor selection algorithm, where choose a individuals between parents and 
            offspring that have great f for the next generation. 
        Arguments:
            parent_f: 1-D numpy array.
            offspring_f: 1-D numpy array.
            features: dictionary where every key is a list of two numpy arrays, which means 
                    key- [parent population feature , offspring population feature].
                Note: This features will be saved to the next generation.
            About:
                If you want more information, check:
                
        ------------------------------------------------------
        """  
        result = {}

        tmp_f = np.concatenate((parent_f,offspring_f))
        indices = get_lowest_indices(tmp_f, len(parent_f))
        result['parent_population_f'] = tmp_f[indices]

        for feature in features.keys():
            if len(features[feature]) != 2:
                raise Exception("Lenght of list have to be 2 (parent population feature and offspring population feature).")
            tmp = np.concatenate((features[feature][0], features[feature][1]), axis=0)
            result[feature] = deepcopy(tmp[indices])
        return result

class replacement_selector:
    def __init__(self):
        self.__doc__ = "Replacement population"

    def __call__(self, parent_f: np.ndarray,\
                       offspring_f: np.ndarray,\
                       features: dict) -> dict:
        """
        ------------------------------------------------------
        Description:
            survivor selection algorithm, where choose m individuals in 
            offspring that have great f for the next generation. 
        Arguments:
            parent_f: 1-D numpy array.
            offspring_f: 1-D numpy array.
            features: dictionary where every key is a list of two numpy arrays, which means 
                    key- [parent population feature , offspring population feature].
                Note: This features will be saved to the next generation.
            About:
                If you want more information, check:
                
        ------------------------------------------------------
        """  
        assert len(parent_f) <= len(offspring_f)
        
        result = {}
        if len(parent_f) < len(offspring_f):
            indices = get_lowest_indices(offspring_f, len(parent_f))
        else:
            indices = range(len(parent_f))

        result['parent_population_f'] = offspring_f[indices]

        for feature in features.keys():
            if len(features[feature]) != 2:
                raise Exception("Lenght of list have to be 2 (parent population feature and offspring population feature).")
            result[feature] = deepcopy(features[feature][1][indices])

        return result