import typing
import numpy as np

class OptimizerConfig:
    """
    Description:
        General configuration for the evolutionary
        optimization algorithms of pyristic.
    Arguments:
        - None
    """
    def __init__(self):
        self.cross_op          = None
        self.mutation_op       = None
        self.survivor_selector = None
        self.fixer             = None

    def cross(self, crossover_: typing.Callable[
                                    [np.ndarray, np.ndarray, np.ndarray],
                                    np.ndarray]):
        """
        Description:
            This method update the current crossover operator
            which the algorithm apply. By default the configuration hasn't
            a crossover operator.
        """
        self.cross_op = crossover_
        return self

    def mutate(self, mutate_ : typing.Callable[[np.ndarray], np.ndarray]):
        """
        Description:
            This method update the current mutator operator
            which the algorithm apply. By default the configuration hasn't
            a mutation operator.
        """
        self.mutation_op = mutate_
        return self

    def survivor_selection(self,
        survivor_function: typing.Callable[
                [np.ndarray,np.ndarray,dict],
                dict
            ]):
        """
        Description:
            This method update the current suvivor selection method
            which the algorithm apply. By default the configuration hasn't
            a survivor method.
        """
        self.survivor_selector = survivor_function
        return self

    def fixer_invalide_solutions(self, fixer_function: typing.Callable[[np.ndarray], np.ndarray]):
        """
        Description:
            This method update the current method to fix the solution
            if this solution doesn't accomplish the constraints
            which the algorithm apply. By default the configuration hasn't
            a fix method.
        """
        self.fixer = fixer_function
        return self
    #-----------------------------------------------------
                    #Private function.
    #-----------------------------------------------------
    def default_printable(self) -> str:
        """
        Description:
            attach the printable information of crossover, mutation, survivor selection.
        """
        printable =\
            "--------------------------------\n\tConfiguration\n--------------------------------\n"
        if self.cross_op is not None:
            printable += "Crossover operator: "+self.cross_op.__doc__ + '\n'
        if self.mutation_op is not None:
            printable += "Mutation operator: "+ self.mutation_op.__doc__ + '\n'
        if self.survivor_selector is not None:
            printable += "Survivor selection: " + self.survivor_selector.__doc__ + '\n'
        if self.fixer is not None:
            printable += "Fixer: " + self.fixer.__doc__ + '\n'
        return printable

class GeneticConfig(OptimizerConfig):
    """
    Description:
        Auxiliar class that keeps the methods need to execute the
        genetic algorithm search. This configuration class store the following methods:
        fixer solutions, parent selection, survivor strategy, crossover operator and
        mutation operator.
    Arguments:
        - None.
    """
    def __init__(self):
        super().__init__()
        self.parent_selector = None

    def __str__(self):
        printable = self.default_printable()
        if self.parent_selector is not None:
            printable += "Parent selection: "+self.parent_selector.__doc__ + '\n'
        printable+="\n--------------------------------"
        return printable

    def parent_selection(self, parent_function: typing.Callable):
        """
        Description:
            It saves in memory the parent selection method.
        Arguments:
            - parent_function: the callback need to perform the parent selection step.
        """
        self.parent_selector = parent_function
        return self

class EvolutionStrategyConfig(OptimizerConfig):
    """
    Description:
        Auxiliar class that keeps the methods need to execute the
        evolutionary strategy algorithm search. This configuration class store the
        following methods: fixer solutions, survivor strategy, crossover operator,
        mutation operator, adaptive mutation and adaptive crossover operator.
    Arguments:
        - None.
    """
    def __init__(self):
        super().__init__()
        self.adaptive_crossover_op = None
        self.adaptive_mutation_op  = None

    def __str__(self):
        printable = self.default_printable()
        if self.adaptive_crossover_op is not None:
            printable += "Adaptive crossover: "+ self.adaptive_crossover_op.__doc__ + '\n'
        if self.adaptive_mutation_op is not None:
            printable += "Adaptive mutation: " + self.adaptive_mutation_op.__doc__ + '\n'
        printable+="\n--------------------------------"
        return printable

    def adaptive_crossover(self, adaptive_crossover_function : typing.Callable):
        """
        Description:
            It saves in memory the adaptive crossover method.
        Arguments:
            - adaptive_crossover_function: the callback need to perform
            the adaptation crossover step.
        """
        self.adaptive_crossover_op = adaptive_crossover_function
        return self

    def adaptive_mutation(self, adaptive_mutation_function: typing.Callable):
        """
        Description:
            It saves in memory the adaptive mutation method.
        Arguments:
            - adaptive_mutation_function: the callback need to perform
            the adaptive mutation step.
        """
        self.adaptive_mutation_op = adaptive_mutation_function
        return self

class EvolutionaryProgrammingConfig(OptimizerConfig):
    """
    Description:
        Auxiliar class that keeps the methods need to execute the
        evolutionary strategy algorithm search. This configuration class store the
        following methods: fixer solutions, survivor strategy, mutation operator,
        adaptive mutation.
    Arguments:
        - None.
    """
    def __init__(self):
        super().__init__()
        self.adaptive_mutation_op  = None

    def __str__(self):
        printable = self.default_printable()
        if self.adaptive_mutation_op is not None:
            printable += "Adaptive mutation: " + self.adaptive_mutation_op.__doc__ + '\n'
        printable+="\n--------------------------------"
        return printable

    def adaptive_mutation(self, adaptive_mutation_function: typing.Callable):
        """
        Description:
            It saves in memory the adaptive mutation method.
        Arguments:
            - adaptive_mutation_function: the callback need to perform
            the adaptive mutation step.
        """
        self.adaptive_mutation_op = adaptive_mutation_function
        return self
