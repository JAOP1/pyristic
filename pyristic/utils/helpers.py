import logging
import io
import time
from tqdm import tqdm
import numpy as np

class TqdmToLogger(io.StringIO):
    """
        Output stream for TQDM which will output to logger module instead of
        the StdOut.
    """
    logger = None
    level = None
    buf = ''
    def __init__(self,logger,level=None):
        super(TqdmToLogger, self).__init__()
        self.logger = logger
        self.level = level or logging.INFO
    def write(self,buf):
        self.buf = buf.strip('\r\n\t ')
    def flush(self):
        self.logger.log(self.level, self.buf)
tqdm_logger = TqdmToLogger(
    logging.getLogger()
)

def get_stats(
    optimizer,
    num_iterations,
    optimizer_args,
    optimizer_additional_args={},
    transformer=None,
    verbose=True,
) -> dict:
    """
    ------------------------------------------------------
    Description:
        Return a dictionary with information about how the opmizer algorithm
        perform after N evaluations applied to opmizer function.

        The dictionary has the mean solution obtained, standard desviation, worst and
        best solution.

    Arguments:
        - optimizer: optimization class.
        - num_iterations: evaluation number, which is the number of times to
        applied class.optimize(args).
        - optimizer_args: Arguments necessary to perform.
        - optimizer_additional_args(optional): additional arguments passed to optimizer.
        - transformer: Function that return a float value. The fuction input is
            the best individual obtained after an execution.
        - verbose: display every solution and objective function.
    ------------------------------------------------------
    """
    data_by_execution = {"execution_time": [], "individual_x": [], "individual_f": []}
    for _ in tqdm(range(num_iterations), file=tqdm_logger, miniters=4,):
        start_time = time.time()
        optimizer.optimize(*optimizer_args, **optimizer_additional_args)
        data_by_execution["execution_time"].append(time.time() - start_time)
        data_by_execution["individual_x"].append(optimizer.logger["best_individual"])
        function_value = optimizer.logger["best_f"]
        if transformer is not None:
            function_value = transformer(optimizer.logger["best_individual"])
        data_by_execution["individual_f"].append(function_value)

    ind_worst = np.argmax(data_by_execution["individual_f"])
    ind_best = np.argmin(data_by_execution["individual_f"])
    stats = {"Worst solution": {}, "Best solution": {}}

    stats["Worst solution"]["x"] = data_by_execution["individual_x"][ind_worst]
    stats["Best solution"]["x"] = data_by_execution["individual_x"][ind_best]

    stats["Worst solution"]["f"] = data_by_execution["individual_f"][ind_worst]
    stats["Best solution"]["f"] = data_by_execution["individual_f"][ind_best]
    stats["Mean"] = np.mean(data_by_execution["individual_f"])
    stats["Standard deviation"] = np.std(data_by_execution["individual_f"])
    stats["Median"] = np.median(data_by_execution["individual_f"])

    if verbose:
        stats.update(data_by_execution)

    return stats


class ContinuosFixer:
    """
    Description:
        When a solution is pass to this callback, the solution is setting as follow:
        Given an interval, values outside the interval are clipped to the interval edges.
        For example, if an interval of [0, 1] is specified, values smaller than 0 become 0,
        and values larger than 1 become 1.
    Arguments:
        - bounds: float list. The first element is the lower bound and
            the second element is the upper bound. Where could be arrays meaning
            the boundaries for every decision variable.
    """

    def __init__(self, bounds: list):
        self.bounds = bounds
        self.__doc__ = "continuos"

    def __call__(self, solution: np.ndarray, ind: int):
        return np.clip(solution[ind], self.bounds[0], self.bounds[1])


class NoneFixer:
    """
    Description:
        This callback doesn't anything. It only keeps as a standard of pyristic.
    Arguments:
        - None.
    """

    def __init__(self):
        self.__doc__ = "None"

    def __call__(self, solution: np.ndarray, ind: int):
        return solution
