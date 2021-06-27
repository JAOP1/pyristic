# Algoritmos Genéticos

La búsqueda se importar:

```python
from pyristic.heuristic.GeneticAlgorithm_search import Genetic
```

!!! info "Genetic (clase de la metaheurística)"

    === "Variables"

        * *__logger.__* Diccionario con información relacionada a la búsqueda.
        
            * `best_individual.` Mejor individuo encontrado.
            * `best_f.` Aptitud del mejor individuo.
            * `current_iter.` Iteración actual de la búsqueda.
            * `total_iter.` Número total de iteraciones.
            * `population_size.` Tamaño de la población.
            * `parent_population_x.` Arreglo bidimensional de numpy. Cada fila representa a un individuo de la población actual y cada columna corresponde a una variable de decisión.
            * `offspring_population_x.` Arreglo bidimensional de numpy. Cada fila representa a un individuo de la población de hijos y cada columna corresponde a una variable de decisión.
            * `parent_population_f.` Arreglo de numpy que contiene el valor de la función objetivo para cada uno de los individuos de la población de `parent_population_x`.
            * `offspring_population_f.` Arreglo de numpy que contiene el valor de la función objetivo para cada uno de los individuos de la población de `offspring_population_x`.
        
        * *__f.__* Función objetivo.
        * *__Constraints.__* Lista de restricciones del problema. Las restricciones deben ser funciones que retornan True o False, indicando si cumple dicha restricción.
        * *__Bounds.__* Representa los límites definidos para cada una de las variables del problema. Se aceptan las siguientes representaciones:
        * Arreglo de *numpy* con solo dos componentes numéricas, donde, la primera componente es el límite inferior y la segunda componente es el límite superior. Esto significa que todas las variables de decisión estarán definidas para el mismo intervalo.
        * Arreglo bidimensional de *numpy* con dos arreglos de numpy, donde, el primer arreglo de numpy representa el límite inferior para cada variable de decisión, mientras, la segunda componente representa el límite superior para cada variable de decisión.
        * *__Decision_variables.__* El número de variables de decisión del problema.

    === "Métodos"

        * *__\__init_\___* Constructor de la clase.
        
            Argumentos:
            
            - `function.` Función objetivo.
            - `decision_variables.` Número de variables de decisión.
            - `constraints.` Lista con las restricciones del problema. Las restricciones deben ser funciones que retornan True o False, dependiendo de si se cumple o no dicha restricción.
            - `bounds.` Límites de las variables de decisión (se describe los tipos de datos admisibles en el apartado de variables de la clase con el nombre Bounds).
            - `config.` Estructura de datos (`GeneticConfig`) con los operadores que se emplearán en la búsqueda.

            Valor de retorno:
            
            - Ninguno.
        
        

        * *__optimize.__* Método principal, realiza la ejecución de la metaheurística.

            Argumentos:
            
            - `generations.` Número de generaciones (iteraciones de la metaheurística).
            - `size_population.` Tamaño de la población (número de individuos).
            - `verbose.` Indica si se imprime en qué iteración se encuentra nuestra búsqueda. Por defecto, está en True.
            - `**kwargs.` Diccionario con argumentos externos a la búsqueda. Estos argumentos pueden ser empleados cuando se sobreescribe alguno de los métodos que tiene la clase.
            
            Valor de retorno:
            
            - Ninguno.

        
        * *__fixer.__*Si nuestro individuo al ser evaluado no cumple las restricciones del problema, esta función auxiliar actualizará nuestro individuo de modo que sea válida. La función auxiliar es necesario definirla (No tiene ninguna por defecto), se especifica en la configuración (GeneticConfig) o sobreescribiendo dicha función.
        
            Argumentos:
            
            - `ind.` Índice del individuo.
                
            Valor de retorno:
            
            - Un arreglo de *numpy* con la solución factible.
        
        * *__initialize_population.__* Crea una población de individuos aleatorios. Para ello se utiliza una distribución uniforme y se generan números aleatorios dentro de los límites indicados para cada variable. Los individuos generados son almacenados en `logger` con la llave `parent_population_x`. Esta función es llamada dentro de la función `optimize`.

            Argumentos:
            
            - `**kwargs.` Diccionario con argumentos externos a la búsqueda. Estos argumentos pueden ser empleados cuando se sobreescribe alguno de los métodos que tiene la clase.
            
            Valor de retorno:
            
            - Un arreglo bidimensional de *numpy*. Cada fila representa un individuo, cada columna indica los valores para cada variable de decisión del individuo.

        * *__mutation_operator.__* Muta las variables de decisión de la población de hijos, las cuales se encuentran almacenadas en el diccionario `logger` con la llave `offspring_population_x`. El operador de mutación es necesario definirlo (No tiene ningún operador por defecto), se especifica en la configuración (`GeneticConfig`) o sobreescribiendo el operador.

            Argumentos:
            
            - `**kwargs.` Diccionario con argumentos externos a la búsqueda. Estos argumentos pueden ser empleados cuando se sobreescribe alguno de los métodos que tiene la clase.
            
            Valor de retorno:
            
            - Un arreglo bidimensional de *numpy* con la población mutada. Cada fila representa un individuo y cada columna corresponde a una variable de decisión.

        * *__crossover_operator.__* Dada una población de padres, genera una población de hijos aplicando algún tipo de cruza. El operador de cruza es necesario definirlo (No tiene ningún operador por defecto), se especifica en la configuración (`GeneticConfig`) o sobreescribiendo el operador.
     
            Argumentos:
            
            - `parent_ind1.` Índices de los individuos que son seleccionados para actuar como padre 1.
            - `parent_ind2.` Índices de los individuos que son seleccionados para actuar como padre 2.
            - `**kwargs.`  Diccionario con argumentos externos a la búsqueda. Estos argumentos pueden ser empleados cuando se sobreescribe alguno de los métodos que tiene la clase.
            
            Valor de retorno:
            
            - Un arreglo bidimensional de *numpy* con la población generada por la cruza. Cada fila representa un individuo y cada columna corresponde a una variable de decisión.

        * *__parent_selection.__* Selecciona a los individuos que actuarán padres. El método de selección es necesario definirlo (No tiene ningún operador por defecto), se especifica en la configuración (`GeneticConfig`) o sobreescribiendo el método.
 
            Argumentos:
            
            - `**kwargs.` Diccionario con argumentos externos a la búsqueda. Estos argumentos pueden ser empleados cuando se sobreescribe alguno de los métodos que tiene la clase.
            
            Valor de retorno:
            
            - Regresa un arreglo con los índices de los individuos seleccionados. Estos índices corresponden con la variable `parent_population_x` que está almacenada en el diccionario `logger`.

        * *__survivor_selection.__* Selección de los individuos que pasarán a la siguiente generación. El método de selección de sobrevivientes es necesario definirlo (No tiene ningún operador por defecto), se especifica en la configuración (`GeneticConfig`) o sobreescribiendo el método.

            Argumentos:
            
            - `**kwargs.` Diccionario con argumentos externos a la búsqueda. Estos argumentos pueden ser empleados cuando se sobreescribe alguno de los métodos que tiene la clase.

            Valor de retorno:
            
            - Un diccionario con las siguientes llaves:
                
                * `parent_population_f.` Arreglo de *numpy* con la aptitud de cada uno de los  individuos que pasará a la siguiente generación.
                * `parent_population_x.` Arreglo bidimensional de *numpy* con los individuos que pasarán a la siguiente generación.
        
## Configuración de algoritmos genéticos

Se debe importar como sigue:
```python
from pyristic.utils.helpers import GeneticConfig
```

!!! info "GeneticConfig (Configuración de la metaheurística)"

    === "Variables"
        * *__cross\_op.__* Variable con el operador de cruza.
        * *__mutation\_op.__* Variable con el operador de mutación.
        * *__survivor\_selection.__* Variable con el esquema de selección que decide cuáles individuos pasan a la siguiente generación.
        * *__fixer.__* Variable con una función que determina qué hacer con los individuos que no cumplen las restricciones del problema.
        * *__parent\_selector.__* Variable que almacena el operador de selección.

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
            
            - Retorna la configuración con la actualización del operador de cruza. El objetivo es poder aplicar varios operadores en cascada.

        * *__survivor_selection.__* Actualiza el operador de selección de la variable `survivor_selector`.

            Argumentos:
            
            - `survivor_function.` Función o clase que realiza la selección de individuos que pasarán a la siguiente generación.

            Valor de retorno:
            
            - Retorna la configuración con la actualización del operador de cruza. El objetivo es poder aplicar varios operadores en cascada.

        * *__fixer\_invalide\_solutions.__* Actualiza la función auxiliar de la variable `fixer`.

            Argumentos:
            
            - `fixer_function.` Función o clase que ajustará los individuos de la población que no cumplen con las restricciones del problema.

            Valor de retorno:
            
            - Retorna la configuración con la actualización del operador de cruza. El objetivo es poder aplicar varios operadores en cascada.

        * *__parent\_selection.__* Actualiza el operador de selección de los individuos con mayores posibilidades de reproducción, se encuentran en la variable `parent_selector`.

            Argumentos:
            
            - `parent_function.` Función o clase que elige los individuos de acuerdo a su contribución de aptitud. Este método en la búsqueda es realizado antes de la cruza.
            
            Valor de retorno:
            
            - Retorna la configuración con la actualización del operador de cruza. El objetivo es poder aplicar varios operadores en cascada.



## Ejemplo
Hay varias formas de emplear la metaheurística, le sugerimos revisar los ejemplos. A continuación presentamos un ejemplo de empleo para el problema de ackley_: 

```python 
configuration = (helpers.GeneticConfig()
                 .cross(crossover.intermediate_crossover(0.5))
                 .mutate(mutation.uniform_mutator(ackley_['bounds']))
                 .survivor_selection(selection.merge_selector())
                 .parent_selection(selection.tournament_sampler(transformer, 3, 0.5))
                 .fixer_invalide_solutions(helpers.ContinuosFixer(ackley_['bounds'])))

AckleyGenetic  = Genetic(**ackley_,config=configuration)
AckleyGenetic.optimize(200,100)
print(AckleyGenetic)
```