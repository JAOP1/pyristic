from pyristic.utils.helpers import *
from random import random 
import math
import numpy as np
import copy 

__all__=['SimulatedAnnealing']

class SimulatedAnnealing:

    def __init__(self, function : function_type,\
                       constraints : list):
        """
        ------------------------------------------------------
        Description:
            Initializing every variable necessary to the search.
        Arguments: 
            - Function: Objective function to minimize.
            - Constraints: Constraints to be a feasible solution.
        ------------------------------------------------------ 
        """
        self.f = function
        self.Constraints = constraints

        #Search information.
        self.logger = {}
        self.logger['best_individual']   = None
        self.logger['best_f']     = None

    def __str__(self):
        printable = "Simulated Annealing: \n f(X) = {} \n X = {} \n ".format(self.logger['best_f'], self.logger['best_individual'])
        first = True

        for i in range(len(self.Constraints)):
            if self.Constraints[i].__doc__ != None:

                if first:
                    first = False
                    printable += "Constraints: \n "

                self.Constraints[i](self.logger['best_individual'])
                printable += "{} \n".format( self.Constraints[i].__doc__)
        return printable
    
    
    def optimize(self,  Init : (np.ndarray,function_type),\
                        IniTemperature : float ,\
                        eps : float,\
                        **kwargs) -> None:
        """
        ------------------------------------------------------
        Description:
            Main function to find the best solution using Simulated Annealing strategy.
        Arguments:
            - Init: Numpy array, represent the initial solution or function which generates a random 
              initial solution (this solution should be a numpy array).
            - IniTemperature:  Floating value, it define how much allow worse solutions.
            - eps: it means what's the final temperature.
        ------------------------------------------------------
        """

        if type(Init) == function_type:
            candidate = Init()
        else:
            candidate = copy.deepcopy(Init)

        self.logger['best_individual'] = copy.deepcopy(candidate)
        self.logger['best_f'] = self.f(copy.deepcopy(candidate))
        self.logger['temperature'] = IniTemperature
        
        f_candidate =self.f(candidate)


        while self.logger['temperature'] >= eps:
            Neighbor = self.get_neighbor(candidate,**kwargs)
            
            if not self.is_valid(Neighbor):
                continue

            f_neighbor = self.f(Neighbor)

            if f_neighbor < f_candidate or random() < self.P(f_neighbor,f_candidate,**kwargs):
                candidate = Neighbor
                f_candidate = f_neighbor
            
            self.logger['temperature'] = self.update_temperature(**kwargs)
            
            #Update best solution found.
            if f_candidate < self.logger['best_f']:
                self.logger['best_f'] = f_candidate
                self.logger['best_individual'] = candidate

        
    def P(self, f_n : float , f_x : float,**kwargs) -> float:
        T = self.logger['temperature']
        return math.exp( -(f_n - f_x)/ T)

    def is_valid(self , x: np.ndarray) -> bool:
        """
        ------------------------------------------------------
        Description:
            Check if the current solution is valid. 
        ------------------------------------------------------
        """
        for constrain in self.Constraints:
            if not constrain(x):
                return False
        return True
    
    def update_temperature(self,**kwargs) -> float:
        """
        ------------------------------------------------------
        Description:

        ------------------------------------------------------
        """
        return  self.logger['temperature'] *0.99

    def get_neighbor(self, x : np.ndarray,**kwargs) -> np.ndarray:
        """
        ------------------------------------------------------
        Description:
           Return only one solution which is a "random" 
            variation of current solution.
        ------------------------------------------------------
        """
        raise NotImplementedError


