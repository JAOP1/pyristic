"""
Module: Evolutionary programming algorithm.

Created: 2023-06-01
Author: Jesus Armando Ortiz
__________________________________________________
"""
import typing
import logging
from tqdm import tqdm
import numpy as np
from pyristic.heuristic.mixins import PrintableMixin, ValidateSolutionMixin
from pyristic.utils.helpers import ContinuosFixer
from pyristic.utils.operators import mutation, selection, population_sample

__all__ = ["EvolutionaryProgramming"]
LOGGER = logging.getLogger()


class EvolutionaryProgramming(PrintableMixin, ValidateSolutionMixin):
    """
    ------------------------------------------------------
    Description:
        Initializing every variable necessary to the search.
    Arguments:
        - Function: Objective function to minimize.
        - Decision variables: Number of decision variables.
        - Constraints: Constraints to be a feasible solution.
        - Bounds: Bounds for each variable. This should be a matrix 2 x N
        where N is the variables number. The first element is the lower limit,
        and another one is the upper limit.
        - Config: EvolutationaryProgrammingConfig, which changes the behavior of
        some operators.
    ------------------------------------------------------
    """

    def __init__(
        self,
        function: typing.Callable[[np.ndarray], typing.Union[int, float]],
        decision_variables: int,
        constraints: list = [],
        bounds: list = [],
        config=None,
    ):
        self.aptitude_function = function
        self.constraints = constraints
        self.bounds = bounds
        self.decision_variables = decision_variables

        # Setting operators.
        self.config_methods = {
            "init_population": population_sample.RandomUniformPopulation(
                self.decision_variables, self.bounds
            ),
            "mutation_operator": mutation.SigmaMutator(),
            "survivor_selector": selection.MergeSelector(),
            "setter_invalid_solution": ContinuosFixer(self.bounds),
            "adaptive_mutation_operator": mutation.SigmaEpAdaptiveMutator(
                self.decision_variables, 0.5
            ),
        }
        if config:
            self.config_methods.update(config.methods)

        # Global information.
        self.logger = {}
        self.logger["best_individual"] = None
        self.logger["best_f"] = None
        self.logger["current_iter"] = None
        self.logger["total_iter"] = None
        self.logger["parent_population_size"] = None
        self.logger["parent_population_x"] = None
        self.logger["parent_population_sigma"] = None
        self.logger["parent_population_f"] = None
        self.logger["offspring_population_x"] = None
        self.logger["offspring_population_sigma"] = None
        self.logger["offspring_population_f"] = None

    def optimize(
        self, generations: int, size_population: int, verbose=True, **_
    ) -> None:
        """
        ------------------------------------------------------
        Description:
            The main function to find the best solution using Evolutionary programming.
        Arguments:
            - Generations: Number of iterations.
            - Population size: Number of individuals.
        ------------------------------------------------------
        """
        generations = int(generations)
        size_population = int(size_population)
        # Reset global solution.
        self.logger["current_iter"] = 0
        self.logger["total_iter"] = generations
        self.logger["parent_population_size"] = size_population
        self.logger["best_individual"] = None
        self.logger["best_f"] = None

        # Initial population.

        self.logger["parent_population_x"] = self.initialize_population(**_)
        self.logger["parent_population_sigma"] = self.initialize_step_weights(**_)
        self.logger["parent_population_f"] = np.apply_along_axis(
            self.aptitude_function, 1, self.logger["parent_population_x"]
        )

        try:
            for _ in tqdm(range(generations), disable=not verbose):
                # Mutation.
                self.logger["offspring_population_sigma"] = self.adaptive_mutation(**_)

                self.logger["offspring_population_x"] = self.mutation_operator(**_)

                # Fixing solutions and getting aptitude.
                f_offspring = []
                for ind in range(len(self.logger["offspring_population_x"])):
                    if self.is_invalid(self.logger["offspring_population_x"][ind]):
                        self.logger["offspring_population_x"][ind] = self.fixer(ind)
                    f_offspring.append(
                        self.aptitude_function(
                            self.logger["offspring_population_x"][ind]
                        )
                    )
                self.logger["offspring_population_f"] = np.array(f_offspring)

                # Survivor selection.
                next_generation = self.survivor_selection(**_)
                self.logger["parent_population_x"] = next_generation[
                    "parent_population_x"
                ]
                self.logger["parent_population_sigma"] = next_generation[
                    "parent_population_sigma"
                ]
                self.logger["parent_population_f"] = next_generation[
                    "parent_population_f"
                ]

                self.logger["current_iter"] += 1

        except KeyboardInterrupt:
            LOGGER.error("Interrupted, saving best solution found so far.")

        ind = np.argmin(self.logger["parent_population_f"])
        self.logger["best_individual"] = self.logger["parent_population_x"][ind]
        self.logger["best_f"] = self.logger["parent_population_f"][ind]

    def mutation_operator(self, **_) -> None:
        """
        ------------------------------------------------------
        Description:
            A new population is created from the current population.
            This function applies small changes to the decision variables
            and the step sizes.
        ------------------------------------------------------
        """
        return self.config_methods["mutation_operator"](
            self.logger["parent_population_x"], self.logger["parent_population_sigma"]
        )

    def adaptive_mutation(self, **_) -> None:
        """
        ------------------------------------------------------
        Description:
            Apply the adaptive mutation operator selected by the configuration.
        ------------------------------------------------------
        """
        return self.config_methods["adaptive_mutation_operator"](
            self.logger["parent_population_sigma"]
        )

    def survivor_selection(self, **_) -> dict:
        """
        ------------------------------------------------------
        Description:
            Apply the survivor selection method selected by the configuration.
        ------------------------------------------------------
        """
        individuals = {}
        individuals["parent_population_x"] = [
            self.logger["parent_population_x"],
            self.logger["offspring_population_x"],
        ]
        individuals["parent_population_sigma"] = [
            self.logger["parent_population_sigma"],
            self.logger["offspring_population_sigma"],
        ]

        return self.config_methods["survivor_selector"](
            self.logger["parent_population_f"],
            self.logger["offspring_population_f"],
            individuals,
        )

    def initialize_step_weights(self, **_) -> np.ndarray:
        """
        ------------------------------------------------------
        Description:
            Create the steps for every individual.
        ------------------------------------------------------
        """
        steps = np.random.uniform(
            0, 1, size=(self.logger["parent_population_size"], self.decision_variables)
        )
        return np.maximum(steps, 0.00001)

    def initialize_population(self, **_) -> np.ndarray:
        """
        ------------------------------------------------------
        Description:
            This function creates a population using a uniform distribution.
            It returns a matrix of size (n x m), where n is the size population
            and m is the number of variables.
        Arguments:
            -n: The population size.
        ------------------------------------------------------
        """
        return self.config_methods["init_population"](
            self.logger["parent_population_size"]
        )

    def fixer(self, ind: int) -> np.ndarray:
        """
        ------------------------------------------------------
        Description:
        This function helps to move one solution to the feasible region.
        Arguments:
            - ind: Index of the individual.
        ------------------------------------------------------
        """
        return self.config_methods["setter_invalid_solution"](
            self.logger["offspring_population_x"], ind
        )
