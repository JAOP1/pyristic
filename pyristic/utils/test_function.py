"""
Module: Classic optimization test problems.
Created: 2023-06-01
Author: Jesus Armando Ortiz
__________________________________________________
"""
import numpy as np


def beale_function(solution: np.ndarray) -> float:
    """
    Beale optimization function.

    Arguments:
        solution: floating numpy array with only two elements.
    """
    return (
        (1.5 - solution[0] + solution[0] * solution[1]) ** 2
        + (2.25 - solution[0] + solution[0] * solution[1] ** 2) ** 2
        + (2.625 - solution[0] + solution[0] * solution[1] ** 3) ** 2
    )


def constraint1_beale(solution: np.ndarray) -> bool:
    """
    Beale's constraints.

    Arguments:
        solution: floating numpy array with only two elements.
    """
    for component in solution:
        if not -4.5 <= component <= 4.5:
            return False
    # If you want to see the result.
    constraint1_beale.__doc__ = (
        f"x1: -4.5 <= {solution[0]:.2f} <= 4.5 \n x2: -4.5 <= {solution[1]:.2f} <= 4.5"
    )
    return True


beale_constraints = [constraint1_beale]
beale_bounds = [-4.5, 4.5]

beale_ = {
    "function": beale_function,
    "constraints": beale_constraints,
    "bounds": beale_bounds,
    "decision_variables": 2,
}

"""
--------------------------------------------------------------------------
                            Ackley function.
--------------------------------------------------------------------------
"""


def ackley_function(solution: np.ndarray) -> float:
    """
    Ackley optimization function.

    Arguments:
        solution: floating numpy array.
    """
    size = len(solution)
    square_sum = (1 / size) * np.sum(solution * solution)
    trigonometric_sum = (1 / size) * np.sum(np.cos(2 * np.pi * solution))

    return (
        -20 * np.exp(-0.2 * np.sqrt(square_sum)) - np.exp(trigonometric_sum) + 20 + np.e
    )


def constraint1_ackley(solution: np.ndarray) -> bool:
    """
    Ackley's constraints.

    Arguments:
        solution: floating numpy array.
    """
    str_ = ""
    valid = True
    for i, element in enumerate(solution):
        if -30 > element or element > 30:
            valid = False
        str_ += f"x{i + 1}: -30 <= {element:.2f} <= 30 \n "

    # Important if you want to see the result.
    constraint1_ackley.__doc__ = str_
    return valid


ackley_constraints = [constraint1_ackley]
ackley_bounds = [-30.0, 30.0]

ackley_ = {
    "function": ackley_function,
    "constraints": ackley_constraints,
    "bounds": ackley_bounds,
    "decision_variables": 10,
}

"""
--------------------------------------------------------------------------
                            Bukin N.6 function.
--------------------------------------------------------------------------
"""


def bukin_function(solution: np.ndarray) -> float:
    """
    Bukin optimization function.

    Arguments:
        solution: floation numpy array.
    """
    return 100.0 * np.sqrt(
        np.abs(solution[1] - 0.01 * solution[0] ** 2)
    ) + 0.01 * np.abs(solution[0] + 10)


def constraint1_bukin(solution: np.ndarray) -> bool:
    """
    Bukin's constraints.

    Arguments:
        solution: floating numpy array.
    """
    str_ = ""
    str_ += f"x{1}: -15 <= {solution[0]:.2f} <= -5 \n "
    str_ += f"x{2}: -3 <= {solution[1]:.2f} <= 3 \n "
    constraint1_bukin.__doc__ = str_
    if -15 > solution[0] or solution[0] > -5:
        return False
    if -3 > solution[1] or solution[1] > 3:
        return False
    return True


bukin_constraints = [constraint1_bukin]
bukin_bounds = [[-15, -3], [-5, 3]]  # [[Lower bounds] , [Upper bounds]]

bukin_ = {
    "function": bukin_function,
    "constraints": bukin_constraints,
    "bounds": bukin_bounds,
    "decision_variables": 2,
}

"""
--------------------------------------------------------------------------
                            Himmelblau's function.
--------------------------------------------------------------------------
"""


def himmelblau_function(solution: np.ndarray) -> float:
    """
    Himmelblau optimization function.

    Arguments:
        solution: floating numpy array.
    """
    return (solution[0] ** 2 + solution[1] - 11) ** 2 + (
        solution[0] + solution[1] ** 2 - 7
    ) ** 2


def constraint1_himmelblau(solution: np.ndarray) -> bool:
    """
    Himmelblau's constraints.

    Arguments:
        solution: floating numpy array.
    """
    str_ = ""
    valid = True
    for ind, decision_var in enumerate(solution):
        if -5 > decision_var or decision_var > 5:
            valid = False
        str_ += f"x{ind+1}: -5 <= {decision_var:.2f} <= 5 \n "

    # Important if you want to see the result.
    constraint1_himmelblau.__doc__ = str_
    return valid


Himmelblau_constraints = [constraint1_himmelblau]
Himmelblau_bounds = [-5, 5]

Himmelblau_ = {
    "function": himmelblau_function,
    "constraints": Himmelblau_constraints,
    "bounds": Himmelblau_bounds,
    "decision_variables": 2,
}
