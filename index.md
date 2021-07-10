# ¡Bienvenido a Pyristic!

Librería de metaheurísticas para resolver problemas de optimización mono-objetivo con o sin restricciones en el lenguaje de python. Los fundamentos de pyristic es proporcionar una accesibilidad para cualquier desarrollador con conocimiento en el área de optimización.

Para iniciar es necesario instalar pyristic. Recomendamos crear un entorno de conda como sigue:
```console
    conda create pyristic-env
    conda activate pyristic-env
    pip install pyristic
```


Para comprobar que todo está funcionando correctamente, crea un script de python `pyristicTest.py` con el siguiente código:

``` python
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

Si todo funciona correctamente es momento de conocer como está estructurada la librería. Actualmente, se encuentran todos los algoritmos en *heuristic* y todas las utilidades en *utils* (funciones prueba, clases para almacenar las configuraciones y los operadores para las metaheurísticas de cómputo evolutivo).

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
