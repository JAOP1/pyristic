import typing
import logging
from tqdm import tqdm
import numpy as np
from pyristic.utils.operators import selection, mutation, crossover, population_sample
from pyristic.utils.helpers import ContinuosFixer

__all__ = ["EvolutionStrategy"]
LOGGER = logging.getLogger()


class EvolutionStrategy:
    """
    ------------------------------------------------------
    Description:
        Initializing every variable necessary to the search.
    Arguments:
        - Function: Objective function to minimize.
        - Constraints: Constraints to be a feasible solution.
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
        self.aptitude_function = function
        self.constraints = constraints
        self.bounds = bounds
        self.decision_variables = decision_variables

        # Configuration.
        self.config_methods = {
            "init_population": population_sample.RandomUniformPopulation(
                self.decision_variables, self.bounds
            ),
            "mutation_operator": mutation.SigmaMutator(),
            "crossover_operator": crossover.DiscreteCrossover(),
            "survivor_selector": selection.MergeSelector(),
            "setter_invalid_solution": ContinuosFixer(self.bounds),
            "adaptive_crossover_operator": crossover.IntermediateCrossover(),
            "adaptive_mutation_operator": mutation.MultSigmaAdaptiveMutator(
                self.decision_variables
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

    def __str__(self):
        printable = (
            "Evolution Strategy search: \n "
            f"F_a(X) = {self.logger['best_f']} \n X = {self.logger['best_individual']} \n "
        )
        first = True

        for constraint in self.constraints:
            if constraint.__doc__:

                if first:
                    first = False
                    printable += "Constraints: \n "

                constraint(self.logger["best_individual"])
                printable += f"{constraint.__doc__} \n"

        return printable

    def optimize(
        self,
        generations: int,
        population_size: int,
        offspring_size: int,
        eps_sigma: float = 0.001,
        verbose=True,
        **kwargs,
    ) -> None:
        """
        ------------------------------------------------------
        Description:
            The main function to find the best solution using tabu search.
        Arguments:
            - generations: integer that represent the total iterations.
            - population_size: the population that
        ------------------------------------------------------
        """
        generations = int(generations)
        population_size = int(population_size)
        offspring_size = int(offspring_size)
        # Reset global solution.
        self.logger["best_individual"] = None
        self.logger["best_f"] = None
        self.logger["current_iter"] = 0
        self.logger["total_iter"] = generations
        self.logger["parent_population_size"] = population_size
        self.logger["offspring_population_size"] = offspring_size
        self.logger["parent_population_x"] = self.initialize_population(**kwargs)
        self.logger["parent_population_sigma"] = self.initialize_step_weights(
            eps_sigma, **kwargs
        )
        self.logger["parent_population_f"] = np.apply_along_axis(
            self.aptitude_function, 1, self.logger["parent_population_x"]
        )

        try:
            for generation in tqdm(range(generations), disable=not verbose):

                # Crossover.
                first_parent_indices, second_parent_indices = self.__get_pairs(**kwargs)
                self.logger["offspring_population_x"] = self.crossover_operator(
                    first_parent_indices, second_parent_indices, **kwargs
                )

                self.logger["offspring_population_sigma"] = self.adaptive_crossover(
                    first_parent_indices, second_parent_indices, **kwargs
                )
                # mutate.
                self.logger["offspring_population_sigma"] = self.adaptive_mutation(
                    **kwargs
                )
                self.logger["offspring_population_x"] = self.mutation_operator(**kwargs)

                self.__set_invalid_individuals()
                next_generation = self.survivor_selection(**kwargs)
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

    def initialize_step_weights(self, eps_sigma: float, **kwargs) -> np.ndarray:
        """
        Description:
            Initialize the size of the steps for every individual.
        """
        steps = np.random.uniform(
            0,
            1,
            size=(
                self.logger["parent_population_size"],
                self.config_methods["adaptive_mutation_operator"].length,
            ),
        )
        return np.maximum(steps, eps_sigma)

    def initialize_population(self, **kwargs) -> np.ndarray:
        """
        ------------------------------------------------------
        Description:
            How to distribute the population. This function create a
            population using uniform distribution.
            This should return an matrix of size (M x N)
            where M is the number of population and N is the number of
            variables.
        Arguments:
            -size_: Tnteger n where n is the number of variables about the problem.
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
            - ind: index of individual.
        ------------------------------------------------------
        """
        return self.config_methods["setter_invalid_solution"](
            self.logger["offspring_population_x"], ind
        )

    def crossover_operator(
        self, parent_ind1: np.ndarray, parent_ind2: np.ndarray, **kwargs
    ) -> np.ndarray:
        """
        ------------------------------------------------------
        Description:
            Apply the crossover operator selected by the configuration.
        ------------------------------------------------------
        """
        return self.config_methods["crossover_operator"](
            self.logger["parent_population_x"], parent_ind1, parent_ind2
        )

    def adaptive_crossover(
        self, parent_ind1: np.ndarray, parent_ind2: np.ndarray, **kwargs
    ) -> np.ndarray:
        """
        ------------------------------------------------------
        Description:
            Apply the adaptive crossover operator selected by the configuration.
        ------------------------------------------------------
        """
        return self.config_methods["adaptive_crossover_operator"](
            self.logger["parent_population_sigma"], parent_ind1, parent_ind2
        )

    def mutation_operator(self, **kwargs) -> np.ndarray:
        """
        ------------------------------------------------------
        Description:
            The current population is updated by specific change
            using the adaptive control.
            This function should mutate the population.
        ------------------------------------------------------
        """
        return self.config_methods["mutation_operator"](
            self.logger["offspring_population_x"],
            self.logger["offspring_population_sigma"],
        )

    def adaptive_mutation(self, **kwargs) -> np.ndarray:
        """
        ------------------------------------------------------
        Description:
            Apply the adaptive mutation operator selected by the configuration.
        ------------------------------------------------------
        """
        return self.config_methods["adaptive_mutation_operator"](
            self.logger["offspring_population_sigma"]
        )

    def survivor_selection(self, **kwargs) -> dict:
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

    # -----------------------------------------------------
    # Private functions.
    # -----------------------------------------------------
    def __is_invalid(self, individual: np.ndarray) -> bool:
        """
        ------------------------------------------------------
        Description:
            Check if the current solution is invalid.
        ------------------------------------------------------
        """
        for constraint in self.constraints:
            if not constraint(individual):
                return True
        return False

    def __get_pairs(self, **kwargs):
        parent_ind1 = np.random.randint(
            self.logger["parent_population_size"],
            size=(self.logger["offspring_population_size"],),
        )
        parent_ind2 = np.random.randint(
            self.logger["parent_population_size"],
            size=(self.logger["offspring_population_size"],),
        )

        return parent_ind1, parent_ind2

    def __set_invalid_individuals(self):
        # Fixing solutions and getting aptitude.
        f_offspring = []
        for i in range(len(self.logger["offspring_population_x"])):
            if self.__is_invalid(self.logger["offspring_population_x"][i]):
                self.logger["offspring_population_x"][i] = self.fixer(i)
            f_offspring.append(
                self.aptitude_function(self.logger["offspring_population_x"][i])
            )

        self.logger["offspring_population_f"] = np.array(f_offspring)
