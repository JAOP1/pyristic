# Funciones auxiliares y funciones de prueba.

## Funciones de prueba
En esta sección se describe los problemas de optimización continua implementados, estos se encuentran en:

```python
import pyristic.utils.test_function as pcTest
```

La representación de las funciones implementadas es en un diccionario con las siguientes llaves:

* `function`: La función a optimizar.
* `constraints`: Restricciones del problema.
* `bounds`: Límites para cada una de las variables del problema. Si nuestro problema todas las variables están en el mismo rango podemos expresarlo en un solo arreglo con el límite inferio y límite superior, sino, deben ser dos arreglos, expresando los límites inferiores y límites superiores.
* `decision_variables`: El número de variables del problema. 

!!! Question "Observaciones"
    
    * Las metaheurísticas descritas de la librería son para resolver problemas de minimización, entonces, se debe convertir el problema, es decir, $max \{ f(x) \}= min \{ -f(x)\}$ 

    * Las llaves señaladas, son importantes al momento de ser llamadas por la metaheurística. Se debe emplear las mismas llaves.

Las funciones prueba que están actualmente son: 

* beale_
* ackley_
* himmelblau_
* bukin_

Para mayor información, puedes revisar las funciones prueba en [wikipedia](https://en.wikipedia.org/wiki/Test_functions_for_optimization).

## Evaluando nuestra metaheurística
Pyristic cuenta con una función auxiliar `get_stats` para evaluar la eficacia de nuestro algoritmo al encontrar soluciones buenas. 

La podemos importar de la siguiente manera:
 
```python
from pyristic.utils.helpers import  get_stats
```

Los parámetros son:

* `optimizer`: Nuestro objeto creado para resolver el problema (metaheurística).
* `NIter`: Número de ejecuciones de nuestro objeto.
* `OptArgs`: Una tupla con los parámetros de nuestro algoritmo (al momento de ejecutar el método *optimize*).
* `ExternOptArgs`: Diccionario con los argumentos incluidos al momento de sobreescribir alguno de los métodos en la metaheurística desarrollada.

### Ejemplo utilizando get_stats

```python
from pyristic.heuristic.EvolutionStrategy_search import EvolutionStrategy
import pyristic.utils.test_function as pcTest
from pyristic.utils.helpers import  get_stats
from pprint import pprint

Beale = EvolutionStrategy(**pcTest.beale_)
args = (200, 80, 160, False)
statistics = get_stats(Beale, 21, args)
pprint(statistics)
```