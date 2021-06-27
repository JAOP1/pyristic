# Estrategias evolutivas 

La búsqueda se importar:

```python
from pyristic.heuristic.EvolutionStrategy_search import EvolutionStrategy
```

!!! info "EvolutionStrategy (clase de la metaheurística)"

    === "Variables"
        * *__logger.__* Diccionario con información relacionada a la búsqueda. 
            - `best_individual.` Individuo con el mejor valor encontrado en la función objetivo.
            - `best_f.` Valor de la función objetivo.
            - `current_iter.` Iteración actual de la búsqueda.
            - `total_iter.` Número total de iteraciones.
            - `parent_population_size.` Tamaño de la población de padres.
            - `offspring_population_size.` Tamaño de la población de hijos.
            - `parent_population_x.` Arreglo bidimensional de *numpy*. Cada fila representa un individuo de la población actual y cada columna corresponde a una variable de decisión.
            - `offspring_population_x.` Arreglo bidimensional de *numpy*. Cada fila representa un individuo de la población de hijos y cada columna corresponde a una variable de decisión.
            - `parent_population_sigma.` Arreglo de *numpy*, donde, cada elemento representa el desplazamiento de todas las variables de decisión de cada individuo o un arreglo bidimensional de *numpy*, donde,  cada fila representa el desplazamiento de cada una de  las variables de decisión de un individuo. 
            - `offspring_population_sigma.`Arreglo de *numpy*, donde, cada elemento representa el desplazamiento de todas las variables de decisión de un individuo o un arreglo bidimensional de *numpy*, donde,  cada fila representa el desplazamiento de cada una de  las variables de decisión de un individuo. 
            - `parent_population_f.` Arreglo de *numpy* que contiene el valor de la función objetivo para cada uno de los individuos de la población con la llave `parent_population_x` en `logger`.
            - `offspring_population_f.` Arreglo de *numpy* que contiene el valor de la función objetivo para cada uno de los individuos de la población con la llave `offspring_population_x`.
        * *__f.__* Función objetivo.
        * *__Constraints.__* Lista de restricciones del problema. Las restricciones deben ser funciones que retornan True o False, indicando si cumple dicha restricción.
        * *__Bounds.__* Representa los límites definidos para cada una de las variables del problema. Se aceptan las siguientes representaciones:
        
            - Arreglo de *numpy* con solo dos componentes numéricas, donde, la primera componente es el límite inferior y la segunda componente es el límite superior. Esto significa que todas las variables de decisión estarán definidas para el mismo intervalo.
            - Arreglo bidimensional de *numpy* con únicamente dos filas, donde, la primer fila es el límite inferior para cada variable de decisión, mientras, la segunda fila representa el límite superior para cada variable de decisión.
        * *__Decision_variables.__* El número de variables de decisión del problema.

    === "Métodos"

        * *__\_\_init\_\_.__* Constructor de la clase.

            Argumentos:
            
            - `function.` Función objetivo.
            - `decision_variables.` Número de variables de decisión del problema.
            - `constraints.` Lista con las restricciones del problema (se describe los tipos de datos admisibles en el apartado de variables de la clase con el nombre `Constraints`).
            - `bounds.` Límites de las variables de decisión (se describe los tipos de datos admisibles en el apartado de variables de la clase con el nombre `Bounds`).  
            - `config.` Estructura de datos (`EvolutionStrategyConfig`) con los operadores que se emplearán en la búsqueda.
            
            Valor de retorno:
            
            - Ninguno.

        * *__optimize.__* Método principal, realiza la ejecución de la metaheurística.

            Argumentos:
            
            - `generations.`  Número de generaciones (iteraciones de la metaheurística).
            - `population_size.` Tamaño de la población (número de individuos).
            - `offspring_size.` Tamaño de la población creada a partir de los operadores de cruza y mutación.
            - `eps_sigma.` Valor mínimo que pueden tener los tamaños de paso. Por defecto, está en 0.001.
            - `verbose.`  Indica si se imprime en qué iteración se encuentra nuestra búsqueda. Por defecto, está en True.
            - `**kwargs.` Diccionario con argumentos externos a la búsqueda. Estos argumentos pueden ser empleados cuando se sobreescribe alguno de los métodos que tiene la clase.
            
            Valor de retorno: 
            
            - Ninguno.
        
        * *__fixer__*. Si la solución no está dentro de los límites definidos para cada variable (restricciones de caja), actualiza el valor de la variable con el valor del límite que rebasó. De lo contrario, regresa la misma solución.

            Argumentos:
            
            - `ind.` Índice del individuo.
            
            Valor de retorno:
            
            - Un arreglo de *numpy* que reemplazará la solución infactible.
        
        * *__initialize_population__*. Crea una población de individuos aleatorios. Para ello se utiliza una distribución uniforme y se generan números aleatorios dentro de los límites indicados para cada variable. Los individuos generados son almacenados en `logger` con la llave `parent_population_x`. Esta función es llamada dentro de la función `optimize`.

            Argumentos:
            
            - `**kwargs.` Diccionario con argumentos externos a la búsqueda. Estos argumentos pueden ser empleados cuando se sobreescribe alguno de los métodos que tiene la clase.
            
            Valor de retorno:
            
            - Un arreglo bidimensional de *numpy*. Cada fila representa un individuo, cada columna indica los valores para cada variable de decisión del individuo.
        
        * *__initialize_step_weights__*. Inicializa el tamaño de desplazamiento de cada individuo de la población, por defecto se emplea un sigma por cada variable de decisión en cada uno de los individuos. Para ello se generan números aleatorios en el intervalo $[0,1]$, utilizando una distribución uniforme. Los tamaños de desplazamiento están almacenados en logger con la llave `parent_population_sigma`.

            Argumentos:
            
            - `eps_sigma.` Valor mínimo que pueden tomar sigma (tamaños de paso).
            - `**kwargs.` Diccionario con argumentos externos a la búsqueda. Estos argumentos pueden ser empleados cuando se sobreescribe alguno de los métodos que tiene la clase.
            
            Valor de retorno:
            
            - Un arreglo bidimensional de *numpy*. Cada fila almacena la información de los tamaños de paso de cada individuo, cada columna pertenece al tamaño de paso de una de las variables de decisión.

        * *__crossover_operator.__* Genera $\lambda$ hijos (individuos nuevos), aplicando una recombinación sexual. Es decir, se seleccionan dos individuos aleatoriamente de `parent_population_x` que actuarán como padres y generarán un hijo. Este procedimiento se repite $\lambda$ veces. Los nuevos individuos se almacenan en `logger` con la llave `offspring_population_x`.
    
            Argumentos:
            
            - `parent_ind1.` Índices de los individuos que son seleccionados para actuar como padre 1.
            - `parent_ind2.` Índices de los individuos que son seleccionados para actuar como padre 2.
            - `**kwargs.` Diccionario con argumentos externos a la búsqueda. Estos argumentos pueden ser empleados cuando se sobreescribe alguno de los métodos que tiene la clase.
            
            Valor de retorno:
            
            - Un arreglo bidimensional de *numpy* con los valores de las variables de decisión de los nuevos individuos.

            *Por defecto la metaheurística utiliza el operador de cruza discreta que se encuentra en `utils.operators.crossover` con el nombre de `discrete`.*

        * *__mutation_operator.__* Muta las variables de decisión de los individuos creados con el operador de cruza. Estos individuos están almacenados en el diccionario `logger` con la llave `offspring_population_x`. La mutación se realiza de la siguiente forma por defecto:

            \begin{equation}
            \label{eq:mutarVariables}
            x'_i = x_i + \sigma'_i \cdot N_i(0, 1)
            \end{equation}

            donde $x'_i$ es la variable mutada, $x_i$ la variable a mutar, $\sigma'_i$ el tamaño de paso (previamente mutado) y $N_{i}(0,1)$ devuelve un número aleatorio por cada variable de decisión utilizando una distribución normal con media $0$ y desviación estándar igual con $1$.

            *Nota*. Es importante tener en cuenta que la fila $j$ de `offspring_population_x` debe corresponder con la fila $j$ de `offspring_population_sigma`.
            
            Argumentos:
            
            - `**kwargs.` Diccionario con argumentos externos a la búsqueda. Estos argumentos pueden ser empleados cuando se sobreescribe alguno de los métodos que tiene la clase.
            
            Valor de retorno:
            
            - Un arreglo bidimensional de numpy que almacena los nuevos valores de las variables de decisión.

        * *__adaptive_crossover.__* Genera los tamaños de paso de los nuevos individuos, aplicando una recombinación sexual. Utiliza las mismas parejas de padres que se usaron con el operador `crossover_operator`. Los nuevos tamaños de paso son almacenados en `logger` con la llave `offspring_population_sigma`.

            Argumentos:
            
            - `parent_ind1.` Índices de los individuos que son seleccionados para actuar como padre 1.
            - `parent_ind2.` Índices de los individuos que son seleccionados para actuar como padre 2.
            - `**kwargs.` Diccionario con argumentos externos a la búsqueda. Estos argumentos pueden ser empleados cuando se sobreescribe alguno de los métodos que tiene la clase.
            
            Valor de retorno:
            
            - Un arreglo bidimensional de *numpy* con los valores de los nuevos tamaños de paso.
            
            *Por defecto la metaheurística utiliza el operador de cruza intermedia que se encuentra en `utils.operators.crossover` con el nombre de `intermediate`.*
        
        * *__adaptive_mutation.__*  Muta los tamaños de paso que se encuentran almacenados en el diccionario `logger` con la llave `offspring_population_sigma`. Este método se ejecuta antes del método `mutation_operator`. La mutación se realiza de la siguiente forma por defecto:

            \begin{equation*}
            \label{eq:mutarNSigmas}
            \sigma'_i = \sigma_i \cdot e ^ {\tau' \cdot N(0,1) + \tau \cdot N_i(0,1)}
            \end{equation*}
        
            Donde,
            
            - $\sigma_{i}$ es el tamaño de paso actual.
            - $\sigma'_{i}$ es tamaño de paso mutado. 
            - $\tau$ está definido como $\frac{1}{\sqrt{2n}}$, donde, $n$ es el número de variables del problema.
            - $\tau'$ está definido como $\frac{1}{\sqrt{2\sqrt{n}}}$, donde, $n$ es el número de variables del problema.
            - $N(0,1)$ devuelve un número aleatorio usando una distribución normal con media 0 y desviación estándar igual a 1. Es importante notar que se genera un único número aleatorio  para todas las $\sigma_{i}$.
            - $N_{i}(0,1)$ devuelve un número aleatorio por $\sigma_i$ utilizando una distribución normal con media 0 y desviación estandas igual a 1.   
            
            Argumentos:
            
            - `**kwargs.` Diccionario con argumentos externos a la búsqueda. Estos argumentos pueden ser empleados cuando se sobreescribe alguno de los métodos que tiene la clase.
            
            Valor de retorno:
            
            - Un arreglo bidimensional de *numpy* con los tamaños de desplazamiento para cada una de las variables de decisión de los individuos.

        * *__survivor_selection.__* Selecciona los individuos que formarán parte de la siguiente generación. 
            
            Argumentos: 
            
            - `**kwargs.` Diccionario con argumentos externos a la búsqueda. Estos argumentos pueden ser empleados cuando se sobreescribe alguno de los métodos que tiene la clase.
            
            Valor de retorno:
            
            - Un diccionario con las siguientes llaves:
                
                * `parent_population_fitness.` El valor de aptitud de cada individuo que pasará a la siguiente generación.
                * `parent_population_sigma.` El/los valor(es) de desplazamiento de los individuos seleccionados.
                * `parent_population_x.` el vector $\vec{x}$ de cada uno de los individuos. 
            
            *Por defecto la metaheurística utiliza el esquema de selección $(\mu + \lambda)$ que se encuentra en `utils.operators.selection` con el nombre de `merge_selector`.*
            

## Configuración de Estrategias evolutivas
Se debe importar como sigue:
```python
from pyristic.utils.helpers import EvolutionStrategyConfig
```
!!! info "EvolutionStrategyConfig (Configuración de la metaheurística)"

    === "Variables"
        * *__cross\_op.__* Variable con el operador de cruza.
        * *__mutation\_op.__* Variable con el operador de mutación.
        * *__survivor\_selector.__* Variable con el esquema de selección que decide cuáles individuos pasan a la siguiente generación.
        * *__fixer.__* Variable con una función que determina qué hacer con los individuos que no cumplen las restricciones del problema. 
        * *__adaptive\_crossover\_op.__* Variable con el operador de cruza que se aplica a los tamaños de paso $\sigma$.
        * *__adaptive\_mutation\_op.__* Variable con el operador de mutación que se aplica a los tamaños de paso $\sigma$.

    === "Métodos"

        * *__cross.__* Actualiza el operador de cruza de la variable `cross_op`.
            
            Argumentos:
            
            - `crossover_.` Función o clase que realiza la cruza de la población almacenada con la llave `parent_population_x`.
            
            Valor de retorno:
            
            - Retorna la configuración con la actualización del operador de cruza. El objetivo es poder aplicar varios operadores en cascada.
            
        * *__mutate.__* Actualiza el operador de mutación de la variable `mutation_op`.

            Argumentos:
            
            - `mutate_.` Función o clase que realiza la mutación de la población almacenada con la llave `offspring_population_x`.
            
            Valor de retorno:
            
            - Retorna la configuración con la actualización del operador de mutación. El objetivo es poder aplicar varios operadores en cascada.
            
        
        * *__survivor\_selection.__* Actualiza el esquema de selección de la variable `survivor_selector`.
        
            Argumentos:
            
            - `survivor_function.` Función o clase que realiza la selección de individuos que formarán parte de la siguiente generación.
            
            Valor de retorno:
            
            - Retorna la configuración con la actualización del esquema de selección de sobrevivientes. El objetivo es poder aplicar varios operadores en cascada.
            
        
        * *__fixer\_invalide\_solutions.__* Actualiza la función de la variable `fixer`.

            Argumentos:
            
            - `fixer_function.` Función o clase que ajustará los individuos de la población que no cumplen con las restricciones del problema.
            
            Valor de retorno:
            
            - Retorna la configuración con la actualización de la función auxiliar. El objetivo es poder aplicar varios operadores en cascada.
            
        
        * *__adaptive\_crossover.__* Actualiza el operador de cruza de los $\sigma$ de la variable `adaptive_crossover_op`.

            Argumentos:
            
            - `adaptive_crossover_function.` Función o clase que cruza los tamaños de paso de los individuos seleccionados para la cruza. Estos tamaños de paso están almacenados en el diccionario `logger` con la llave `offspring_population_sigma`. 
            
            Valor de retorno:
            
            - Retorna la configuración con la actualización del operador de cruza en los tamaños de paso. El objetivo es poder aplicar varios operadores en cascada.
            
        
        * *__adaptive\_mutation.__* Actualiza el operador de mutación de los $\sigma$ de la variable `adaptive_mutation_op`.

            Argumentos:
            
            - `adaptive_mutation_function.` Función o clase que muta los tamaños de paso que se encuentran en `logger`con la llave `offspring_population_sigma`.
            
            Valor de retorno:
            
            - Retorna la configuración con la actualización del operador de mutación en los tamaños de paso. El objetivo es poder aplicar varios operadores en cascada.

## Ejemplo

Hay varias formas de emplear la metaheurística, le sugerimos revisar los ejemplos. A continuación presentamos un ejemplo de empleo para el problema de ackley_: 

```python 
configuration_ackley = (EvolutionStrategyConfig()
                       .survivor_selection(selection.replacement_selector())
                       .adaptive_mutation(
                           mutation.single_sigma_adaptive_mutator(
                                           ackley_['decision_variables'])
                                         )
                       )
solver_ackley_custom = EvolutionStrategy(**ackley_,config=configuration_ackley)
solver_ackley_custom.optimize(250,100,200)
print(solver_ackley_custom)
```
