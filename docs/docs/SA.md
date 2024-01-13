# Recocido simulado 

La búsqueda de recocido simulado la podemos importar:

```python
from pyristic.heuristic import SimulatedAnnealing
```

!!! info "SimulatedAnnealing (Clase recocido simulado)"

    === "Variables"

        * *__logger.__* Diccionario con información relacionada a la búsqueda con las siguientes llaves:
            * `best_individual.` Mejor individuo encontrado.
            * `best_f.` El valor obtenido de la función objetivo de `individual`.
            * `temperature.` Temperatura inicial que se actualizará cada iteración.
        * *__f.__* Función objetivo.
        * *__Constraints.__* Lista de restricciones del problema. Las restricciones deben ser funciones que retornan True o False, indicando si cumple dicha restricción.

    === "Métodos"

        * *__\_\_init\_\_.__* Inicializa la clase.

            Argumentos:

            - `function.` Función objetivo.
            - `constraints.` Lista con las restricciones del problema.

            Valor de retorno:
        
            - Ninguno.


        * *__optimize.__* método principal, realiza la ejecución empleando la metaheurística llamada `SimulatedAnnealing`.

            Argumentos:

            - `Init.` Solución inicial, se admite un arreglo de *numpy* o una función que retorne un arreglo de *numpy*.
            - `IniTemperature.` Valor de punto flotante que indica con que temperatura inicia la búsqueda.
            - `eps.` Valor de punto flotante que indica con que temperatura termina la búsqueda.
            - `**_.` Parámetros externos a la búsqueda.
        
            Valor de retorno:
        
            - Ninguno

        * *__update\_temperature.__* Función que decrementa la temperatura.

            Argumentos:
            
            - `**_` Parámetros externos a la búsqueda.
            
            Valor de retorno:
            
            - La nueva temperatura.

        !!! warning "Funciones que se deben sobreescribir"
            * *__get_neighbor.__* Genera una solución realizando una variación aleatoria en la solución actual.

                Argumentos:
                
                - `x.` Arreglo de *numpy* representando a la solución actual.
                - `**_` Parámetros externos a la búsqueda.
                
                Valor de retorno:
                
                - Arreglo de *numpy* representando la solución generada.
            
            

## Ejemplo
Para emplear la búsqueda se debe crear una clase nueva que herede todos los métodos de `SimulatedAnnealing`, por ejemplo:

```python

    class simulatedAnnealingExample(SimulatedAnnealing):

        def __init__(self, f_ : function_type , constraints_: list):
            super().__init__(f_,constraints_)

        def get_neighbor(self, x : np.ndarray) -> np.ndarray:             
            # Código para su búsqueda.


```