# Programación evolutiva

La búsqueda se importar:

```python
from pyristic.heuristic import EvolutionaryProgramming
```

!!! info "EvolutionaryProgramming (clase de la metaheurística)"

    === "Variables"

        * *__logger.__* Diccionario con información relacionada a la búsqueda con las siguientes llaves:

            * `best_individual.` Mejor individuo encontrado.
            * `best_f.` El vlor obtenido de la función objetivo evaluado en el individuo almacenado en individual.
            * `current_iter.` Iteración actual de la búsqueda.
            * `total_iter.` Número total de iteraciones.
            * `parent_population_x.` Arreglo bidimensional de numpy que representa cada fila a un individuo de la población, mientras, las columnas representan el número de variables de decisión.
            * `offspring_population_x.` Arreglo bidimensional de numpy que representa cada fila a un individuo de la población, mientras, las columnas representan el número de variables de decisión.
            * `parent_population_sigma.` Arreglo de numpy que representa el desplazamiento de por variable de decisión de cada uno de los individuos.
            * `offspring_population_sigma.` Arreglo de numpy que representa el desplazamiento de por variable de decisión de cada uno de los individuos.
            * `parent_population_f.` Arreglo de numpy que contiene el valor de la función objetivo para cada uno de los individuos de la población de parent_population_x.
            * `offspring_population_f.` Arreglo de numpy que contiene el valor de la función objetivo para cada uno de los individuos de la población de offspring_population_x.

        * *__f.__* Función objetivo.
        * *__Constraints.__* Lista de restricciones del problema. Las restricciones deben ser funciones que retornan True o False, indicando si cumple dicha restricción.
        * *__Bounds.__*  Representa los límites definidos para cada una de las variables del problema. Se aceptan las siguientes representaciones:

            * Arreglo de numpy con solo dos componentes numéricas, donde, la primera componente es el límite inferior y la segunda componente es el límite superior. Esto significa que todas las variables de decisión estarán definidas para el mismo intervalo.
            * Arreglo bidimensional de numpy con dos arreglos de numpy, donde, el primer arreglo de numpy representa el límite inferior para cada variable de decisión, mientras, la segunda componente representa el límite superior para cada variable de decisión.
            
        * *__Decision\_variables.__* El número de variables de decisión del problema.

    === "Métodos"

        * *__\_\_init\_\_.__* Constructor de la clase.

            Argumentos:

            - `function.` Función objetivo.
            - `decision_variables.` Número que indica las variables de decisión del problema.
            - `constraints.` Lista con las restricciones del problema.
            - `bounds.` Límites del espacio de búsqueda de cada una de las variables de decisión del problema.
            - `config.` Estructura de datos (`EvolutionaryProgrammingConfig`) con los operadores que se emplearán en la búsqueda.

            Valor de retorno:
        
            - Ninguno.


        * *__optimize.__* método principal, realiza la ejecución de la metaheurística.
        
            Argumentos:
            
            - `generations.` Número de generaciones (iteraciones de la metaheurística).
            - `size_population.` Tamaño de la población (número de individuos).
            - `verbose.` Indica si se imprime en qué iteración se encuentra nuestra búsqueda. Por defecto, está en True.
            - `**_.` Diccionario con argumentos externos a la búsqueda. Estos argumentos pueden ser empleados cuando se sobreescribe alguno de los métodos que tiene la clase.
            
            Valor de retorno:
            
            - Ninguno.
        
        
        * *__mutatio\_operator.__*  Muta las variables de decisión que se encuentran almacenadas en el diccionario `logger` con la llave `parent_population_x`.
            
            Argumentos:
            
            - `**_` Diccionario con argumentos externos a la búsqueda. Estos argumentos pueden ser empleados cuando se sobreescribe alguno de los métodos que tiene la clase.
                
            Valor de retorno:
            
            - Un arreglo bidimensional de *numpy* representado a los nuevos individuos (se almacenarán en `logger` con la llave `offspring_population_x`).
                
            
        * *__adaptive\_mutation.__* Muta los tamaños de paso que se encuentran almacenados en el diccionario `logger` con la llave `parent_population_sigma`.

            Argumentos:
            
            - `**_.` Diccionario con argumentos externos a la búsqueda. Estos argumentos pueden ser empleados cuando se sobreescribe alguno de los métodos que tiene la clase.
            
            Valor de retorno:
            
            - Un arreglo bidimensional de numpy con los tamaños de desplazamiento para cada una de las variables de decisión de los individuos (se almacenarán en `logger` con la llave `offspring_population_sigma`).


        * *__survivor\_selection.__* Selección de los individuos que pasarán a la siguiente generación.

            Argumentos:
            
            - `**_.` Diccionario con argumentos externos a la búsqueda. Estos argumentos pueden ser empleados cuando se sobreescribe alguno de los métodos que tiene la clase.
            
            Valor de retorno:
            
            - Un diccionario con las siguientes llaves:
                
                - `parent_population_f.` El valor de la función objetivo de cada individuo que pasará a la siguiente generación.
                - `parent_population_sigma.` El/los valor(es) de desplazamiento de los individuos seleccionados.
                - `parent_population_x.` el vector $\vec{x}$ de cada uno de los individuos.

            Por defecto la metaheurística utiliza el esquema de selección $(\mu + \lambda)$ que se encuentra en utils.operators.selection con el nombre de merge_selector.


        * *__initialize\_population.__* Crea una población de individuos aleatorios. Para ello se utiliza una distribución uniforme y se generan números aleatorios dentro de los límites indicados para cada variable. Los individuos generados son almacenados en `logger` con la llave `parent_population_x`. Esta función es llamada dentro de la función optimize.

            Argumentos:
            
            - `**_.` Diccionario con argumentos externos a la búsqueda. Estos argumentos pueden ser empleados cuando se sobreescribe alguno de los métodos que tiene la clase.
            
            Valor de retorno:
            
            - Un arreglo bidimensional de *numpy*. El número de filas es igual al tamaño de la población y el número de columnas es igual al número de variables que tiene el problema que se está resolviendo.

            
        * *__initialize\_step\_weights.__* Inicializa el tamaño de desplazamiento de cada una de las variables de decisión pertenecientes a cada individuo de la población. Para ello se generan números aleatorios en el intervalo $[0,1]$, utilizando una distribución uniforme. Los tamaños de desplazamiento están almacenados en `logger` con la llave `parent_population_sigma`.

            Argumentos:
            
            - `**_.` Diccionario con argumentos externos a la búsqueda. Estos argumentos pueden ser empleados cuando se sobreescribe alguno de los métodos que tiene la clase.
            
            Valor de retorno:
            
            - Un arreglo bidimensional de numpy. El número de filas es igual al tamaño de la población y el número de columnas es igual al número de variables que tiene el problema que se está resolviendo. Cada variable tiene su propio tamaño de paso.
        
        
        * *__fixer.__* Si la solución no está dentro de los límites definidos para cada variable (restricciones de caja), actualiza el valor de la variable con el valor del límite que rebasó. De lo contrario, regresa la misma solución.
        
            Argumentos:
            
            - `ind.` Índice del individuo.
            
            Valor de retorno:
        
            - Un arreglo de *numpy* que reemplazará la solución infactible de la población con la llave `offspring_population_x`.

## Configuración de programación evolutiva
Se debe importar como sigue:
```python
from pyristic.utils.helpers import EvolutionaryProgrammingConfig 
```

!!! info "EvolutionaryProgrammingConfig (Configuración de la metaheurística)"

    === "Variables"
        * *__mutation\_op.__* Variable con el operador de mutación.
        * *__survivor\_selector.__* Variable con el esquema de selección de los individuos que pasan a la siguiente generación.
        * *__fixer.__* Variable con una función auxiliar para los individuos que no cumplen las restricciones del problema.
        * *__adaptive\_mutation\_op.__* Variable con el operador de mutación de los tamaños de paso $\sigma$.

    === "Métodos"

        * *__mutate.__* Actualiza el operador de mutación de la variable `mutation_op`.

            Argumentos:
            
            - `mutate_.` Función o clase que realiza la mutación de la población almacenada con la llave `offspring_population_x`.

            Valor de retorno:
            
            - Retorna la configuración con la actualización del operador de cruza. El objetivo es poder aplicar varios operadores en cascada.


        * *__survivor\_selection.__* Actualiza el esquema de selección de la variable `survivor_selector`.

            Argumentos:
            
            - `survivor_function.`Función o clase que realiza la selección de individuos para la próxima generación.

            Valor de retorno:
            
            - Retorna la configuración con la actualización del operador de cruza. El objetivo es poder aplicar varios operadores en cascada.

        * *__fixer\_invalide\_solutions.__* Actualiza la función auxiliar de la variable `fixer`.

            Argumentos:
            
            - `fixer_function.` Función o clase que ajustará los individuos de la población que no cumplen con al menos una de las restricciones del problema.

            Valor de retorno:
            
            - Retorna la configuración con la actualización del operador de cruza. El objetivo es poder aplicar varios operadores en cascada.

        * *__adaptive\_mutation.__* Actualiza el operador de mutación de los $\sigma$ de la variable `adaptive_mutation_op`.

            Argumentos:
            
            - `adaptive_mutation_function.` Función o clase que muta los tamaños de paso que se encuentran en `logger` con la llave `parent_population_sigma`.

            Valor de retorno:
            
            - Retorna la configuración con la actualización del operador de cruza. El objetivo es poder aplicar varios operadores en cascada.

## Ejemplo

Hay varias formas de emplear la metaheurística, le sugerimos revisar los ejemplos. A continuación presentamos un ejemplo de empleo para el problema de ackley_: 

```python 
configuration = (EvolutionaryProgrammingConfig()
                 .adaptive_mutation(
                     mutation.sigma_ep_adaptive_mutator(ackley_['decision_variables'], 2.0)
                     ) 
                 )
Optimizer_by_configuration = EvolutionaryProgramming(**ackley_, config=configuration)
Optimizer_by_configuration.optimize(500,100)
```


