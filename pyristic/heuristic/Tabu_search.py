from pyristic.utils.helpers import *
from tqdm import tqdm
import numpy as np
import copy 

__all__=['TabuList','TabuSearch']

class TabuList:
    def __init__(self, timer: int=2):
        self._TB = [[None,None,-1, 100000000]]
        self.timer = timer
        
    def __str__(self):
        printable_object = "\n ---- Tabu list: ---- \n"
        printable_object += "List size: {} \n".format(len(self._TB))
        for i in range(len(self._TB)):
            a,b,c,d = self._TB[i][0], self._TB[i][1], self._TB[i][2], self._TB[i][3]
            printable_object += \
            "p: {} v:{} Iteration: {} current timer: {} \n".format(a,b,c,d)
        return printable_object
        
    def push(self, x: list) -> None:
        assert len(x) == 3
        x.append(self.timer)
        self._TB.insert(0,x)
    
    def pop_back(self) -> None:
        x_last = self._TB.pop()
        x_current_back = self.get_back()
        update_time = self.timer - (x_current_back[2]-x_last[2])
        x_current_back[2] -= update_time
        self.update_back(x_current_back)
        
    def get_back(self) -> list:
        return self._TB[-1]
    
    def update_back(self, x : list) -> None:
        self._TB[-1] = x

    def find(self, x : list) -> bool:
        #X is [p, v], where p is the position changed and v the new value.
        assert len(x) == 2
        for i in range(len(self._TB)):
            if x[0] == self._TB[i][0] and x[1] == self._TB[i][1]:
                return True
        return False

    def reset(self, timer : int) -> None:
        self._TB = [] 
        self.timer = timer
    
    def update(self) -> None:
        most_old_candidate = copy.deepcopy(self.get_back())
        most_old_candidate[-1] -= 1
           
        if(most_old_candidate[-1] <= 0): 
            self.pop_back()
        else:
            self.update_back(most_old_candidate)

class TabuSearch:
 
    def __init__(self,  function: function_type,\
                        constraints : list,\
                        TabuStruct = TabuList()):
        """
        Arguments: 
            - Function: Objective function to minimize.
            - Constraints: Constraints to be a feasible solution.
            - TabuStruct: Structure to store information about search by default is TabuList. 
        Note: 
            The argument in TabuList is not essential. It is changed when you call optimize.
        """
        self.TL = TabuStruct #Initialize tabulist
        self.Constraints = constraints
        self.f = function
        
        #Search information.
        self.logger = {}
        self.logger['best_individual']   = None
        self.logger['best_f']     = None
        self.logger['current_iter'] = None
        self.logger['total_iter']   = None
           
    def __str__(self):
        printable = "Tabu search: \n f(X) = {} \n X = {} \n ".format(self.logger['best_f'], self.logger['best_individual'])
        first = True
        
        for i in range(len(self.Constraints)):
            if self.Constraints[i].__doc__ != None:
                
                if first:
                    first = False
                    printable += "Constraints: \n "
                    
                self.Constraints[i](self.logger['best_individual'])
                printable += "{} \n".format( self.Constraints[i].__doc__)
                   
        return printable
        
    def optimize(self,Init: (np.ndarray,function_type) ,iterations: int,\
                 memory_time : int, verbose:bool=True, **kwargs)->None:
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
        self.TL.reset(memory_time)
        
        self.logger['best_individual'] = None
        self.logger['best_f']   = None
        
        
        if type(Init) == function_type:
            best_candidate = Init()
        else:
            best_candidate = copy.deepcopy(Init)
        
        f_candidate =self.f(best_candidate)
        
        try:
            for step_ in tqdm(range(1,iterations+1), disable=not verbose):

                Neighbors =[neighbor for  neighbor in self.get_neighbors(best_candidate,**kwargs) \
                            if not self.TL.find(self.encode_change(neighbor, best_candidate,**kwargs))]
                Neighbors = np.array(Neighbors)

                ValidNeighbors =  \
                Neighbors[np.apply_along_axis(self.is_valid,1,Neighbors),:] 

                #Check if there exists a valid neighbor
                if len(ValidNeighbors) == 0:
                    continue

                f_feasible_candidates =  \
                np.apply_along_axis(self.f , 1, ValidNeighbors)      

                ind_min = np.argmin(f_feasible_candidates)           

                p,v = self.encode_change(ValidNeighbors[ind_min],best_candidate,**kwargs)
                self.TL.push([p, v, step_])

                if f_feasible_candidates[ind_min]< f_candidate:
                    best_candidate = copy.deepcopy(ValidNeighbors[ind_min])
                    f_candidate = f_feasible_candidates[ind_min]

                self.TL.update()

        except KeyboardInterrupt:
            print("Interrupted, saving best solution found so far.")
            
        self.logger['best_individual'] = best_candidate
        self.logger['best_f'] = f_candidate
    
    def is_valid(self, x : np.ndarray) -> bool:
        """
        ------------------------------------------------------
        Description:
            Check if the current solution is valid. 
        ------------------------------------------------------
        """
        for constraint in self.Constraints:
            if not constraint(x):
                return False
        return True
    
    def get_neighbors(self, x : (list,np.ndarray),\
                            **kwargs)-> list:
        """
        ------------------------------------------------------
        Description:
            The user has to custom this function. 
            The function has to return a list with all possible neighbors. 
            Check the examples for a detailed explanation.
        ------------------------------------------------------
        """
        raise NotImplementedError
            
    def encode_change(self, neighbor: (list,np.ndarray),\
                            x: (list,np.ndarray),\
                            **kwargs) -> list:
        """
        ------------------------------------------------------
        Description: 
            The user has to custom this function.
            Check the examples for a detailed explanation.
        ------------------------------------------------------
        """
        raise NotImplementedError
     