import typing
import numpy as np

__all__ = [
    "OptimizerConfig",
    "GeneticConfig",
    "EvolutionStrategyConfig",
    "EvolutionaryProgrammingConfig",
]


class OptimizerConfig:
    """
    Description:
        General configuration for the evolutionary
        optimization algorithms of pyristic.
    Arguments:
        - None
    """

    def __init__(self):
        self.methods = {}

    def cross(
        self,
        crossover_: typing.Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    ):
        """
        Description:
            This method update the current crossover operator
            which the algorithm apply. By default the configuration hasn't
            a crossover operator.
        """
        self.methods["crossover_operator"] = crossover_
        return self

    def mutate(self, mutate_: typing.Callable[[np.ndarray], np.ndarray]):
        """
        Description:
            This method update the current mutator operator
            which the algorithm apply. By default the configuration hasn't
            a mutation operator.
        """
        self.methods["mutation_operator"] = mutate_
        return self

    def survivor_selection(
        self, survivor_function: typing.Callable[[np.ndarray, np.ndarray, dict], dict]
    ):
        """
        Description:
            This method update the current suvivor selection method
            which the algorithm apply. By default the configuration hasn't
            a survivor method.
        """
        self.methods["survivor_selector"] = survivor_function
        return self

    def fixer_invalide_solutions(
        self, fixer_function: typing.Callable[[np.ndarray], np.ndarray]
    ):
        """
        Description:
            This method update the current method to fix the solution
            if this solution doesn't accomplish the constraints
            which the algorithm apply. By default the configuration hasn't
            a fix method.
        """
        self.methods["setter_invalid_solution"] = fixer_function
        return self

    def initialize_population(self, init_function: typing.Callable):
        """
        Description:
            This function update the current method to initialize the population
            of our algorithm. By default the configuration hasn't a method.
        """
        self.methods["init_population"] = init_function
        return self

    def __str__(self):
        """
        Description:
            attach the printable information of crossover, mutation, survivor selection.
        """
        printable = "--------------------------------\n\tConfiguration\n--------------------------------\n"
        for key, func in self.methods.items():
            printable += f"{key} - {func.__doc__}\n"
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

    def parent_selection(self, parent_function: typing.Callable):
        """
        Description:
            It saves in memory the parent selection method.
        Arguments:
            - parent_function: the callback need to perform the parent selection step.
        """
        self.methods["parent_selector"] = parent_function
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

    def adaptive_crossover(self, adaptive_crossover_function: typing.Callable):
        """
        Description:
            It saves in memory the adaptive crossover method.
        Arguments:
            - adaptive_crossover_function: the callback need to perform
            the adaptation crossover step.
        """
        self.methods["adaptive_crossover_operator"] = adaptive_crossover_function
        return self

    def adaptive_mutation(self, adaptive_mutation_function: typing.Callable):
        """
        Description:
            It saves in memory the adaptive mutation method.
        Arguments:
            - adaptive_mutation_function: the callback need to perform
            the adaptive mutation step.
        """
        self.methods["adaptive_mutation_operator"] = adaptive_mutation_function
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

    def adaptive_mutation(self, adaptive_mutation_function: typing.Callable):
        """
        Description:
            It saves in memory the adaptive mutation method.
        Arguments:
            - adaptive_mutation_function: the callback need to perform
            the adaptive mutation step.
        """
        self.methods["adaptive_mutation_operator"] = adaptive_mutation_function
        return self
