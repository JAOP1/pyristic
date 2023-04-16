import typing
import logging
from tqdm import tqdm
import numpy as np
import copy
from pyristic.utils.helpers import *

__all__ = ["TabuList", "TabuSearch"]
LOGGER = logging.getLogger()


class TabuList:
    """
    Description:
        Complementary data structure to help the
        tabu search, where save the updates to the current
        solution during a specific time.
    Arguments:
        - timer: integer number (default 2). It indicates
            the time that a item is retained in the tabu list.
    """

    def __init__(self, timer: int = 2):
        self.tabu_list = [[None, None, -1, 100000000]]
        self.timer = timer

    def __str__(self):
        printable_object = "\n ---- Tabu list: ---- \n"
        printable_object += f"List size: {self.tabu_list} \n"
        for position, value, iteration, current_timer in self.tabu_list:
            printable_object += (
                f"p: {position} v:{value} Iteration: "
                f"{iteration} current timer: {current_timer} \n"
            )
        return printable_object

    def push(self, solution: list) -> None:
        """
        Description:
            Append a solution during a time in the tabu list.
        Arguments:
            - solution:
        """
        assert len(solution) == 3
        solution.append(self.timer)
        self.tabu_list.insert(0, solution)

    def pop_back(self) -> None:
        """
        Description:
            remove a item from the tabu list.
        Arguments:
            - None
        """
        x_last = self.tabu_list.pop()
        x_current_back = self.get_back()
        update_time = self.timer - (x_current_back[2] - x_last[2])
        x_current_back[2] -= update_time
        self.update_back(x_current_back)

    def get_back(self) -> list:
        """
        Description:
            Return the older element in the list.
        Arguments:
            - None.
        """
        return self.tabu_list[-1]

    def update_back(self, solution: list) -> None:
        """
        Description:
            Update the older solution in the list.
        Arguments:
            - None.
        """
        self.tabu_list[-1] = solution

    def find(self, solution: list) -> bool:
        """
        Description:
            Search in the elem
        Arguments:
            - solution: it is an array of two elements. The first position
            is the location in the current solution and the second position is
            the value replaced.
        """
        # X is [p, v], where p is the position changed and v the new value.
        assert len(solution) == 2
        for position, value, *_ in self.tabu_list:
            if solution[0] == position and solution[1] == value:
                return True
        return False

    def reset(self, timer: int) -> None:
        """
        Description:
            Clean up the data structure.
        Arguments:
            - None.
        """
        self.tabu_list = []
        self.timer = timer

    def update(self) -> None:
        """
        Description:
            Update the data structure, it controls if the older solution
            should be in the container yet.
        Arguments:
            - None.
        """
        most_old_candidate = copy.deepcopy(self.get_back())
        most_old_candidate[-1] -= 1

        if most_old_candidate[-1] <= 0:
            self.pop_back()
        else:
            self.update_back(most_old_candidate)


class TabuSearch:
    """
    ------------------------------------------------------
    Description:
        Optimization search algorithm inspired on Tabu
        algorithm.
    Arguments:
        - Function: The aptitude function to minimize.
        - Constraints: Array with boolean functions.
        - tabu_struct: Include the structure that helps you save relevant
            informationn during the search (optional argument).
    ------------------------------------------------------
    """

    def __init__(
        self,
        function: typing.Callable[[np.ndarray], typing.Union[int, float]],
        constraints: list,
        tabu_struct=TabuList(),
    ):
        self.tabu_list = tabu_struct  # Initialize tabulist
        self.constraints = constraints
        self.function = function

        # Search information.
        self.logger = {}
        self.logger["best_individual"] = None
        self.logger["best_f"] = None
        self.logger["current_iter"] = None
        self.logger["total_iter"] = None

    def __str__(self):
        printable = (
            "Tabu search: \n "
            f"f(X) = {self.logger['best_f']} \n X = {self.logger['best_individual']} \n "
        )
        first = True

        for constraint in self.constraints:
            if constraint.__doc__ is not None:

                if first:
                    first = False
                    printable += "Constraints: \n "

                constraint(self.logger["best_individual"])
                printable += f"{constraint.__doc__} \n"

        return printable

    def optimize(
        self,
        initial_solution: typing.Union[np.ndarray, typing.Callable[[], np.ndarray]],
        iterations: int,
        memory_time: int,
        verbose: bool = True,
        **kwargs,
    ) -> None:
        """
        ------------------------------------------------------
        Description:
            The main function to find the best solution using tabu search.
        Arguments:
            - Init: Numpy array, represent the initial solution or function which generates a random
              initial solution (this solution should be a numpy array).
            - Iterations:  Integer, stop condition.
            - Memory time: Integer, time which a solution is in the tabu list
        ------------------------------------------------------
        """
        try:
            x_candidate = initial_solution()
        except TypeError:
            x_candidate = copy.deepcopy(initial_solution)

        f_candidate = self.function(x_candidate)
        self.tabu_list.reset(memory_time)
        self.logger["best_individual"] = np.copy(x_candidate)
        self.logger["best_f"] = f_candidate
        try:
            for step in tqdm(range(1, iterations + 1), disable=not verbose):

                neighbors = [
                    neighbor
                    for neighbor in self.get_neighbors(x_candidate, **kwargs)
                    if not self.tabu_list.find(
                        self.encode_change(neighbor, x_candidate, **kwargs)
                    )
                ]
                neighbors = np.array(neighbors)
                try:
                    neighbors = neighbors[
                        np.apply_along_axis(self.is_valid, 1, neighbors), :
                    ]
                    f_candidates = np.apply_along_axis(
                        self.function, 1, neighbors
                    )

                    ind_min = np.argmin(f_candidates)

                    position, value = self.encode_change(
                        neighbors[ind_min], x_candidate, **kwargs
                    )
                    self.tabu_list.push([position, value, step])
                    x_candidate = neighbors[ind_min]
                    f_candidate = f_candidates
                    if f_candidates[ind_min] < self.logger["best_f"]:
                        self.logger["best_individual"] = copy.deepcopy(neighbors[ind_min])
                        self.logger["best_f"] = f_candidates[ind_min]
                except ValueError:
                    pass
                self.tabu_list.update()

        except KeyboardInterrupt:
            LOGGER.error("Interrupted, saving best solution found so far.")

    def is_valid(self, solution: np.ndarray) -> bool:
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

    def get_neighbors(self, solution: typing.Union[list, np.ndarray], **kwargs) -> list:
        """
        ------------------------------------------------------
        Description:
            The user has to custom this function.
            The function has to return a list with all possible neighbors.
            Check the examples for a detailed explanation.
        ------------------------------------------------------
        """
        raise NotImplementedError

    def encode_change(
        self,
        neighbor: typing.Union[list, np.ndarray],
        solution: typing.Union[list, np.ndarray],
        **kwargs,
    ) -> list:
        """
        ------------------------------------------------------
        Description:
            The user has to custom this function.
            Check the examples for a detailed explanation.
        ------------------------------------------------------
        """
        raise NotImplementedError
