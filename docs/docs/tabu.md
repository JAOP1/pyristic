# Búsqueda Tabú

La búsqueda tabú la podemos importar:
```python
from pyristic.heuristic import TabuSearch
```


!!! info "TabuSearch (clase búsqueda tabú)"

    === "Variables"
        * *__logger.__* Diccionario con información relacionada a la búsqueda con las siguientes llaves:
            * `best_individual.` Mejor individuo encontrado.
            * `best_f.` El valor obtenido de la función objetivo de `individual`.
            * `current_iter.` Iteración actual de la búsqueda.
            * `total_iter.` Número total de iteraciones.
        * *__TL.__* Estructura de datos auxiliar que mantendrá memoria de las soluciones encontradas durante el tiempo especificado en `optimize`, por defecto utiliza `TabuList`.
        * *__f.__* Función objetivo.
        * *__Constraints.__* Lista de restricciones del problema. Las restricciones deben ser funciones que retornan True o False, indicando si cumple dicha restricción.

    === "Métodos"
        * *__\_\_init\_\_.__* Constructor de la clase.

            Argumentos:

            - `function.` Función objetivo.
            - `constraints.` Lista con las restricciones del problema.
            - `TabuStruct.` Estructura de datos que almacena información de variaciones que mejoran la solución.
            
            Valor de retorno:

            - Ninguno.


        * *__optimize.__* método principal, realiza la ejecución empleando la metaheurística llamada `TabuSearch`.

            Argumentos:
            
            - `Init.` Solución inicial, se admite un arreglo de *numpy* o una función que retorne un arreglo de *numpy*.
            - `iterations.` Número de iteraciones.
            - `memory_time.` Tiempo que permanecerá una solución en nuestra estructura llamada `TabuList`.
            - `**_.` Parámetros externos a la búsqueda.
                
            Valor de retorno:
            - Ninguno

        !!! warning "Funciones que se deben sobreescribir"
            * *__get_neighbors.__* Función que genera el vecindario de soluciones de la solución $x$.

                Argumentos:

                - `x.` Arreglo de *numpy* representando a la solución actual.
                - `**_` Parámetros externos a la búsqueda.
                    
                Valor de retorno:

                - Arreglo bidimensional de *numpy* representando a todas las soluciones generadas desde la solución $x$.
                

            * *__encode_change.__* Revisa nuestra solución actual $x$ y la solución generada para indicar en dónde sucedió la pequeña variación. 

                Argumentos:

                - `neighbor.` Arreglo de *numpy* representando una variación de nuestra solución actual $x$.
                - `x.` Arreglo de *numpy* representando nuestra solución actual.
                - `**_.` Parámetros externos a la búsqueda.
                
                Valor de retorno:
                    
                - Lista con dos elementos, donde, la primera componente será la posición $i$ donde sucedió la variación y la segunda componente es el elemento en la componente $i$ de `neighbor`.

## Lista Tabú

!!! info "TabuList (clase lista tabú)"

    === "Variables"
        * *__\_TB.__* Lista de listas que representarán las posiciones que fueron modificadas con un contador de tiempo.
        * *__timer.__* El tiempo que durará cada solución en la lista.

    === "Métodos"

        * *__push.__* Introduce los cambios que proporcionaron una mejora en la búsqueda.
        
            Argumentos:
            
            - `x.` Arreglo con la siguiente información: 
                * Primera componente: posición (indice) donde se encontró una mejora en la función objetivo.
                * Segunda componente: valor por el cual mejoró nuestra solución.
                * Tercera componente: iteración en la que se realizó la mejora.
        
            Valor de retorno:
            
            - Ninguno.
        
        
        * *__find.__*  Revisa si la nueva solución sea una de las modificaciones hechas en iteraciones previas almacenadas en `_TB`. 
        
            Argumentos:
        
            - `x.` Arreglo de *numpy* que representa el cambio realizado en la solución actual de la búsqueda, es decir, recibe el arreglo que retorna la función `encode_change(neighbor,x)`.  
        
            Valor de retorno:
        
            - Valor booleano que indica si la modificación en dicha solución ya se encontraba en nuestra lista tabú. 
        
        
        * *__reset.__* Borra toda la información almacenada en nuestro contenedor `_TB` y actualiza la variable `timer`.

            Argumentos:
        
            - `timer.` Número que representa el tiempo que durarán ahora las soluciones en nuestra lista tabú.
        
            Valor de retorno:
            
            - Ninguno.
        
        
        * *__update.__* Realiza la actualización en el contendor `_TB` modificando el tiempo de cada uno de los individuos almacenados y elimina aquellos individuos que ya expiró su tiempo. 

            Argumentos:
            
            - Ninguno.
        
            Valor de retorno:
            
            - Ninguno.
        
        
        * *__pop\_back.__*  Elimina el último elemento del contenedor `_TB`.

            Argumentos:
        
            - Ninguno.
        
            Valor de retorno:
            
            - Ninguno.
        
        
        * *__get\_back.__* Regresa el último elemento del contenedor `_TB`.

            Argumentos:
            
            - Ninguno.
            
            Valor de retorno:
            
            - Elemento del contendor `_TB`.


## Ejemplo 
Para emplear la búsqueda tabú se debe crear una clase nueva que herede todos los métodos de `TabuSearch`, por ejemplo:

```python

    class tabuExample(TabuList):

        def __init__(self, f_ : function_type , constraints_: list, TabuStruct_):
            super().__init__(f_, constraints_, TabuStruct)

        def get_neighbors(self, x : np.ndarray,**_) -> list:
            # Código para su búsqueda.

        def encode_change(self, neighbor : (list,np.ndarray), x : (list,np.ndarray),**_) -> list:
            # Código para su búsqueda

```