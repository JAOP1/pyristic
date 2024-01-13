"""
Module: Simulated Annealing algorithm.

Created: 2023-06-01
Author: Jesus Armando Ortiz
__________________________________________________
"""
import typing
from random import random
import math
import copy
import numpy as np
from pyristic.heuristic.mixins import PrintableMixin, ValidateSolutionMixin

__all__ = ["SimulatedAnnealing"]


class SimulatedAnnealing(PrintableMixin, ValidateSolutionMixin):
    """
    ------------------------------------------------------
    Description:
        Optimization search algorithm inspired on simulated
        annealing algorithm.
    Arguments:
        - Function: The aptitude function to minimize.
        - Constraints: Array with boolean functions.
    ------------------------------------------------------
    """

    def __init__(
        self,
        function: typing.Callable[[np.ndarray], typing.Union[int, float]],
        constraints: list,
        neighbor_generator: typing.Callable[[np.ndarray], np.ndarray] = None,
    ):
        self.function = function
        self.constraints = constraints
        self.neighbor_generator = neighbor_generator
        # Search information.
        self.logger = {}
        self.logger["best_individual"] = None
        self.logger["best_f"] = None

    def optimize(
        self,
        initial_solution: typing.Union[np.ndarray, typing.Callable[[], np.ndarray]],
        initial_temperature: float,
        eps: float,
        **_,
    ) -> None:
        """
        ------------------------------------------------------
        Description:
            Main function to find the best solution using Simulated Annealing strategy.
        Arguments:
            - initial_solution: Numpy array, represent the initial solution or function
                which generates a random initial solution (this solution should be a numpy array).
            - initial_temperature:  Floating value, it define how much allow worse solutions.
            - eps: it means what's the final temperature.
        ------------------------------------------------------
        """
        try:
            candidate = initial_solution()
        except TypeError:
            candidate = copy.deepcopy(initial_solution)

        self.logger["best_individual"] = copy.deepcopy(candidate)
        self.logger["best_f"] = self.function(copy.deepcopy(candidate))
        self.logger["temperature"] = initial_temperature

        f_candidate = self.function(candidate)
        while self.logger["temperature"] >= eps:
            neighbor = self.get_neighbor(candidate, **_)

            if not self._is_valid(neighbor):
                continue

            f_neighbor = self.function(neighbor)

            if f_neighbor < f_candidate or random() < self.acceptance_measure(
                f_neighbor, f_candidate, **_
            ):
                candidate = neighbor
                f_candidate = f_neighbor

            self.logger["temperature"] = self.update_temperature(**_)

            # Update best solution found.
            if f_candidate < self.logger["best_f"]:
                self.logger["best_f"] = f_candidate
                self.logger["best_individual"] = candidate

    def acceptance_measure(self, f_n: float, f_x: float, **_) -> float:
        """
        Description:
            Metric used to evaluate how much confidence accept
            a solution.
        """
        temperature = self.logger["temperature"]
        return math.exp(-(f_n - f_x) / temperature)

    def update_temperature(self, **_) -> float:
        """
        ------------------------------------------------------
        Description:

        ------------------------------------------------------
        """
        return self.logger["temperature"] * 0.99

    def get_neighbor(self, solution: np.ndarray, **_) -> np.ndarray:
        """
        ------------------------------------------------------
        Description:
           Return only one solution which is a "random"
            variation of current solution.
        ------------------------------------------------------
        """
        if not self.neighbor_generator:
            raise NotImplementedError
        return self.neighbor_generator(solution)

    def _is_valid(self, solution: np.ndarray) -> bool:
        """
        ------------------------------------------------------
        Description:
            Check if the current solution is valid.
        ------------------------------------------------------
        """
        for constraint in self.constraints:
            if not constraint(solution):
                return False
        return True
