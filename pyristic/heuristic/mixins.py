"""
Module: Mixin helpers for heuristics.

Created: 2023-06-01
Author: Jesus Armando Ortiz
__________________________________________________
"""
import numpy as np


class PrintableMixin:
    """Mixin that provide solution string."""

    def __str__(self):
        """Display the configuration of our metaheuristic."""
        printable = (
            "Heuristic solution: \n "
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


class ValidateSolutionMixin:
    """Mixin that check if the solution is valid."""

    def is_invalid(self, individual: np.ndarray) -> bool:
        """Check if the current solution is invalid."""
        for constraint in self.constraints:
            if not constraint(individual):
                return True
        return False
