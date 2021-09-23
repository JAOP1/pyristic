# Pyristic
Librería de *Python* con metaheurísticas para resolver problemas de optimización mono-objetivo con o sin restricciones. El objetivo de esta librería es facilitar la resolución de problemas de optimización utilizando metaheurísticas. *Pyristic* se divide como sigue:
```
🗀 Pyristic
│   
└─── 🗀 heuristic
│   │   EvolutionStrategy_search 
│   │   EvolutiveProgramming_search
│   │   GeneticAlgorithm_search
│   │   SimulatedAnnealing_search
│   │   Tabu_search
│  
└─── 🗀 utils
     │   helpers
     │   test_function
     │
     └─── 🗀 operators
     │   │   crossover
     │   │   mutation
     │   │   selection 
```


## Instalación
La instalación de la librería y dependencias se realiza a través del manejador de paquetes *pip*:
```
pip install pyristic
```

## Ejemplo

Uso de la librería *Pyristic* para resolver la función de Beale con la metaheurística de Estrategias Evolutivas.
```python
from pyristic.heuristic.EvolutionStrategy_search import EvolutionStrategy
from pyristic.utils.test_function import beale_
"""
Instancia de la clase EvolutionStrategy: 
- Función objetivo (recibe la funci
- Lista de restricciones
- Límite inferior y superior de las variables de decisión
- Número de variables de decisión
"""
BealeOptimizer = EvolutionStrategy(**beale_)

"""
Ejecución de la metaheurística con los siguientes parámetros:
- Número de iteraciones
- Tamaño de la población a cada iteración
- Tamaño de la población de hijos
- Mostrar la iteración en la que se encuentra
"""
BealeOptimizer.optimize(300,80,160,verbose=True)

#Resultados obtenidos por la ejecución del método optimize.
print(BealeOptimizer)
```


## Contribución
Los usuarios interesados en participar deben seguir los siguientes pasos:
1. Clonar el proyecto.
```
git clone https://github.com/JAOP1/pyristic.git
```
2. Crear branch en relación al tipo de acción a realizar (añadir metaheurística, crear utilidad o resolver errores):
   * Solucionar error, *fix-archivo-funcion*.
    ```
    git checkout -b fix-crossover-n_point_crossover
    ```
   * Añadir metaheurística, *attach-metaheuristic-nombreMetaheuristica*. 
   ```
   git checkout -b attach-metaheuristic-fireflyAlgorithm
   ```
   * Crear utilidad, *attach-utility-nombreUtilidad*.
   ```
   git checkout -b attach-utility-binaryCrossover
   ```
3. Realizar commit con un mensaje explicando lo realizado. Por ejemplo:
```
git add pyristic/utils/operators/crossover.py
git commit -m "Operador de cruza para problemas discretos."
```

**Nota:** las metaheurísticas anexadas deben ser clases que mantienen los mismos parámetros en el método \_\_init\_\_ que son:
* function           -> Función objetivo (función de python).
* decision_variables -> Número de variables de decisión (valor entero).
* constraints        -> Restricciones, arreglo con funciones de python que retornan un valor booleano.
* bounds -> Lista con los límites de las variables de decisión.
  Además, debe tener el método *optimize* (con los parámetros de la respectiva metaheurística).
   
### Agradecimientos
* Dra. Adriana Menchaca Méndez (usuario github: [adriana1304](https://github.com/adriana1304)), titular del proyecto que supervisó y evaluó el desarrollo de la librería pyristic.
* El apoyo del Programa UNAM-DGAPA-PAPIME PE102320.
