import inspect
import numpy as np
import time

def f(x):
    pass
function_type = type(f)

def checkargs(function):
    def _f(*arguments):
        for index, argument in enumerate(inspect.getfullargspec(function)[0]):
            
            if argument == 'self':
                continue

            if not isinstance(arguments[index], function.__annotations__[argument]):
                raise TypeError("{} is not of type {}".format(arguments[index], function.__annotations__[argument]))
        return function(*arguments)
    _f.__doc__ = function.__doc__
    return _f


def get_stats(optimizer, NIter, OptArgs,**ExternOptArgs) -> dict:
    """
    ------------------------------------------------------
    Description:
        Return a dictionary with information about how the opmizer algorithm
        perform after N evaluations applied to opmizer function.

        The dictionary has the mean solution obtained, standard desviation, worst and 
        best solution.

    Arguments:
        - optimizer: optimization class.
        - NIter: evaluation number, which is the number of times to 
        applied class.optimize(args).
        - OptArgs: Arguments necessary to perform.
        - ExternOptArgs(optional): additional arguments.
    ------------------------------------------------------
    """
    f_ = []
    x_ = []
    timeByExecution = []
    for i in range(NIter):
        start_time = time.time()
        optimizer.optimize(*OptArgs,**ExternOptArgs)
        timeByExecution.append(time.time() - start_time)
        f_.append(optimizer.logger['best_f'])
        x_.append(optimizer.logger['best_individual'])
        
    IndWorst = np.argmax(f_)
    IndBest = np.argmin(f_)
    stats_ = {"Worst solution":{} , "Best solution":{}}
    
    stats_["Worst solution"]["x"] = x_[IndWorst]
    stats_["Best solution"]["x"] = x_[IndBest]
    
    stats_["Worst solution"]["f"] = f_[IndWorst]
    stats_["Best solution"]["f"] = f_[IndBest]

    stats_["objectiveFunction"] = f_
    stats_["Mean"] = np.mean(f_)
    stats_["averageTime"] = np.mean(timeByExecution)
    stats_["Standard deviation"] = np.std(f_)
    stats_['Median'] = np.median(f_)

    return stats_


# -----------------------------------------------------------------------
#                   Configuration by Evolutionary computing 
# -----------------------------------------------------------------------

class OptimizerConfig:
    def __init__(self):
        self.cross_op          = None
        self.mutation_op       = None
        self.survivor_selector = None 
        self.fixer             = None

    def cross(self, crossover_: function_type):
        self.cross_op = crossover_
        return self

    def mutate(self, mutate_ : function_type):
        self.mutation_op = mutate_
        return self
    
    def survivor_selection(self, survivor_function: function_type):
        self.survivor_selector = survivor_function
        return self

    def fixer_invalide_solutions(self, fixer_function: function_type):
        self.fixer = fixer_function
        return self
    #-----------------------------------------------------
                    #Private function.
    #----------------------------------------------------- 
    def default_printable(self) -> str:
        printable = "--------------------------------\n\tConfiguration\n--------------------------------\n"
        if self.cross_op != None:
            printable += "Crossover operator: "+self.cross_op.__doc__ + '\n'
        if self.mutation_op != None:
            printable += "Mutation operator: "+ self.mutation_op.__doc__ + '\n'
        if self.survivor_selector != None:
            printable += "Survivor selection: " + self.survivor_selector.__doc__ + '\n'
        if self.fixer != None:
            printable += "Fixer: " + self.fixer.__doc__ + '\n'
        return printable

class GeneticConfig(OptimizerConfig):
    def __init__(self):
        super().__init__()
        self.parent_selector = None

    def __str__(self):
        printable = self.default_printable()
        if self.parent_selector != None:
            printable += "Parent selection: "+self.parent_selector.__doc__ + '\n'
        printable+="\n--------------------------------"
        return printable
    
    def parent_selection(self, parent_function: function_type):
        self.parent_selector = parent_function
        return self

class EvolutionStrategyConfig(OptimizerConfig):
    def __init__(self):
        super().__init__()
        self.adaptive_crossover_op = None
        self.adaptive_mutation_op  = None

    def __str__(self):
        printable = self.default_printable()        
        if self.adaptive_crossover_op != None:
            printable += "Adaptive crossover: "+ self.adaptive_crossover_op.__doc__ + '\n'
        if self.adaptive_mutation_op != None:
            printable += "Adaptive mutation: " + self.adaptive_mutation_op.__doc__ + '\n'
        printable+="\n--------------------------------"
        return printable

    def adaptive_crossover(self, adaptive_crossover_function : function_type):
        self.adaptive_crossover_op = adaptive_crossover_function
        return self
    
    def adaptive_mutation(self, adaptive_mutation_function: function_type):
        self.adaptive_mutation_op = adaptive_mutation_function
        return self

class EvolutionaryProgrammingConfig(OptimizerConfig):
    def __init__(self):
        super().__init__()
        self.adaptive_mutation_op  = None

    def __str__(self):
        printable = self.default_printable()        
        if self.adaptive_mutation_op != None:
            printable += "Adaptive mutation: " + self.adaptive_mutation_op.__doc__ + '\n'
        printable+="\n--------------------------------"
        return printable    

    def adaptive_mutation(self, adaptive_mutation_function: function_type):
        self.adaptive_mutation_op = adaptive_mutation_function
        return self         

class ContinuosFixer:
    def __init__(self, bounds: list):
        self.Bounds = bounds
        self.__doc__ = "continuos"

    def __call__(self, X_: np.ndarray, ind: int):
        return np.clip(X_[ind], self.Bounds[0], self.Bounds[1])

class NoneFixer:
    def __init__(self):
        self.__doc__ = "None"

    def __call__(self, X_: np.ndarray, ind: int):
        return X_
        