# Pyristic
Librería de metaheurísticas para resolver problemas de optimización mono-objetivo  con o sin restricciones en el lenguaje de python. Los fundamentos de pyristic es proporcionar una accesibilidad para cualquier desarrollador con conocimiento en el área de optimización.  

Pyristic se divide como sigue:
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

## Ejemplo

Uso de la librería de pyristic para encontrar una solución a un problema continuo con la metaheurística Estrategias evolutivas.
```python
from pyristic.heuristic.EvolutionStrategy_search import EvolutionStrategy
from pyristic.utils.test_function import beale_
"""
Declaración del algoritmo: 
- Función objetivo
- Lista de restricciones
- Límites del problema
- Variables de decisión
"""
BealeOptimizer = EvolutionStrategy(**beale_)

"""
Ejecución de la metaheurística con los siguientes parámetros:
- Número de iteraciones
- Población en cada iteración
- Población de individuos generados por iteración
- Mostrar la iteración en la que se encuentra
"""
BealeOptimizer.optimize(300,80,160,verbose=True)

#Resultados obtenidos por la ejecución del método optimize.
print(BealeOptimizer)
```


## Instalación
La instalación de la librería y dependencias es empleando el manejador de paquetes pip. La instrucción para instalar es la siguiente:

```
pip3 install pyristic
```


## Contribución
Los usuarios interesados en participar deben seguir los siguientes pasos:
1. Clonar el proyecto.
```
git clone https://github.com/JAOP1/pyristic.git
```
2. Crear branch en relación al tipo de acción a realizar (añadir metaheurística, crear utilidad o resolver errores):
   * Solucionar error, *fix-archivo-funcion*. Supongamos que se quiere solucionar un error en algún operador de cruza, se debe realizar como sigue:
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
* decision_variables -> variables de decisión (valor entero).
* constraints        -> Restricciones, arreglo con funciones de python que retornan un valor booleano.
* bounds -> lista con los límites del problema.

  Además, debe tener el método *optimize* (con los parámetros de la respectiva metaheurística).
   
### Agradecimientos
* Dra. Adriana Menchaca Méndez (usuario github: [adriana1304](https://github.com/adriana1304)), titular del proyecto que supervisó y evaluó el desarrollo de la librería pyristic.
* El apoyo del Programa UNAM-DGAPA-PAPIME PE102320.