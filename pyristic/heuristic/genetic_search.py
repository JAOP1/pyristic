"""
Module: Genetic algorithm.

Created: 2023-06-01
Author: Jesus Armando Ortiz
__________________________________________________
"""
import typing
import logging
from tqdm import tqdm
import numpy as np
from pyristic.heuristic.mixins import (
    PrintableMixin,
    ValidateSolutionMixin,
    SaveValidSolutionsMixin,
)
from pyristic.utils.operators import population_sample

__all__ = ["Genetic"]
LOGGER = logging.getLogger()


class Genetic(PrintableMixin, ValidateSolutionMixin, SaveValidSolutionsMixin):
    """
    ------------------------------------------------------
    Description:
        Optimization search algorithm inspired on genetic
        algorithms from evolutionary algorithms field.
    Arguments:
        - Function: The aptitude function to minimize.
        - Constraints: Array with boolean functions.
        - Bounds: bound for every variable, this should be a matrix 2 x N
            where N is the variables number. The first element is lowert limit
            and another one is the upper limit.
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
        # Information about problem.
        self.aptitude_function = function
        self.constraints = constraints
        self.bounds = bounds
        self.decision_variables = decision_variables  # Decision variables.

        # Operators.
        self.config_methods = {
            "init_population": population_sample.RandomUniformPopulation(
                self.decision_variables, self.bounds
            ),
        }
        if config:
            self.config_methods.update(config.methods)
        # Search information.
        self.logger = {}
        self.logger["best_individual"] = None
        self.logger["best_f"] = None
        self.logger["current_iter"] = None
        self.logger["total_iter"] = None
        self.logger["parent_population_size"] = None

    def optimize(
        self,
        generations: int,
        size_population: int,
        cross_percentage: float = 1.0,
        mutation_percentage: float = 1.0,
        verbose: bool = True,
        **_,
    ) -> None:
        """
        ------------------------------------------------------
        Description:
            The main function to find the best solution using tabu search.
        Arguments:
            -generations: number of iterations.
            -population: Number of new solutions.
        ------------------------------------------------------
        """
        assert 0 <= cross_percentage <= 1.0
        assert 0 <= mutation_percentage <= 1.0
        generations = int(generations)
        size_population = int(size_population)
        # Reset global information.
        self.logger["current_iter"] = 0
        self.logger["total_iter"] = generations
        self.logger["parent_population_size"] = size_population
        self.logger["cross_percentage"] = cross_percentage
        self.logger["mutation_percentage"] = mutation_percentage
        self.logger["best_f"] = None
        self.logger["best_individual"] = None

        # Initial population.
        self.logger["parent_population_x"] = self.initialize_population(**_)
        self.logger["parent_population_f"] = np.apply_along_axis(
            self.aptitude_function, 1, self.logger["parent_population_x"]
        )

        try:
            for _ in tqdm(range(generations), disable=not verbose):
                # Parent selection.
                parent_ind = self.parent_selection(**_)
                first_parent_indices, second_parent_indices = self.__get_pairs(
                    parent_ind
                )
                # Cross.
                self.__cross_individuals(first_parent_indices, second_parent_indices)
                # mutate.
                self.__mutate_individuals()

                self.set_invalid_individuals()

                # Survivor selection.
                next_generation = self.survivor_selection(**_)
                self.logger["parent_population_x"] = next_generation["population"]
                self.logger["parent_population_f"] = next_generation[
                    "parent_population_f"
                ]

                self.logger["current_iter"] += 1

        except KeyboardInterrupt:
            LOGGER.error("Interrupted, saving best solution found so far.")

        ind = np.argmin(self.logger["parent_population_f"])
        self.logger["best_individual"] = self.logger["parent_population_x"][ind]
        self.logger["best_f"] = self.logger["parent_population_f"][ind]

    def initialize_population(self, **_) -> np.ndarray:
        """
        ------------------------------------------------------
        Description:
            How to distribute the population. This function create a
            population using uniform distribution.
            This should return an matrix of size (M x N)
            where M is the number of population and N is the number of
            variables.
        Arguments:
            -size_: tuple with two integers (m,n)
            where m is the number generated of new solutions and
            n is the number of variables about the problem.
        ------------------------------------------------------
        """
        return self.config_methods["init_population"](
            self.logger["parent_population_size"]
        )

    def fixer(self, ind: int) -> np.ndarray:
        """
        ------------------------------------------------------
        Description:
        Function which helps to move solution in a valid region.
        Arguments:
            -X: numpy array, this one is the current population.
            -ind: index of individual.
        ------------------------------------------------------
        """
        if "setter_invalid_solution" not in self.config_methods:
            raise NotImplementedError
        return self.config_methods["setter_invalid_solution"](
            self.logger["offspring_population_x"], ind
        )

    def mutation_operator(self, indices, **_):
        """
        ------------------------------------------------------
        Description:
            Apply the mutation operator selected by the configuration.
        Arguments:
            - Indices: the indices (or boolean array) of individuals in offspring_population_x
            to mutate.
        ------------------------------------------------------
        """
        if "mutation_operator" not in self.config_methods:
            raise NotImplementedError
        return self.config_methods["mutation_operator"](
            self.logger["offspring_population_x"][indices]
        )

    def crossover_operator(self, parent_ind1: np.ndarray, parent_ind2: np.ndarray, **_):
        """
        ------------------------------------------------------
        Description:
            Apply the crossover operator selected by the configuration.
        ------------------------------------------------------
        """
        if "crossover_operator" not in self.config_methods:
            raise NotImplementedError
        return self.config_methods["crossover_operator"](
            self.logger["parent_population_x"], parent_ind1, parent_ind2
        )

    def survivor_selection(self, **_) -> dict:
        """
        ------------------------------------------------------
        Description:
            Apply the selection survivors method by the configuration.
        ------------------------------------------------------
        """
        if "survivor_selector" not in self.config_methods:
            raise NotImplementedError

        individuals = {}
        individuals["population"] = [
            self.logger["parent_population_x"],
            self.logger["offspring_population_x"],
        ]

        return self.config_methods["survivor_selector"](
            self.logger["parent_population_f"],
            self.logger["offspring_population_f"],
            individuals,
        )

    def parent_selection(self, **_) -> np.ndarray:
        """
        ------------------------------------------------------
        Description:
            Apply the parent selection method by the configuration.
        ------------------------------------------------------
        """
        if "parent_selector" not in self.config_methods:
            raise NotImplementedError
        return self.config_methods["parent_selector"](
            self.logger["parent_population_f"]
        )

    def __get_pairs(self, parent_ind: np.ndarray):
        parent_ind1 = []
        parent_ind2 = []
        size_ = len(parent_ind) // 2
        ind_individuals_selected = np.random.randint(
            0, len(parent_ind), size=(size_, 2)
        )
        for ind_individual_a, ind_individual_b in ind_individuals_selected:
            parent_ind1.append(parent_ind[ind_individual_a])
            parent_ind2.append(parent_ind[ind_individual_b])

        return parent_ind1, parent_ind2

    def __cross_individuals(
        self, first_parents: np.ndarray, second_parents: np.ndarray
    ) -> None:
        offspring = None
        for parent_a, parent_b in zip(first_parents, second_parents):
            individuals = self.logger["parent_population_x"][[parent_a, parent_b]]

            if np.random.rand() <= self.logger["cross_percentage"]:
                individuals = self.crossover_operator(
                    np.array([parent_a]), np.array([parent_b])
                )
            try:
                offspring = np.concatenate((offspring, individuals), axis=0)
            except ValueError:
                offspring = individuals

        self.logger["offspring_population_x"] = offspring

    def __mutate_individuals(self):
        individuals, _ = self.logger["offspring_population_x"].shape
        individuals_to_mutate = [
            np.random.rand() <= self.logger["mutation_percentage"]
            for i in range(individuals)
        ]
        self.logger["offspring_population_x"][
            individuals_to_mutate
        ] = self.mutation_operator(individuals_to_mutate)
