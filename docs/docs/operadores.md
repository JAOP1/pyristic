# Operadores de cómputo evolutivo 

## Operadores de cruza
Ubicación de operadores:
```python
import pyristic.utils.operators.crossover as pcCross
```

!!! info "Operadores por metaheurística"

    === "Estrategias evolutivas"
        !!! Success ""
            **discrete**. Operador de cruza discreta. Sean $\vec{x}$ y $\vec{y}$ los padres 1 y 2, respectivamente. El nuevo individuo $\vec{z}$ está dado por:
                \begin{equation}
                    z_{i} = 
                    \begin{cases}
                        x_{i} & \text{Si } b_{i} = 1, \\
                        y_{i}  & \text{Si } b_{i} = 0 \\
                    \end{cases}
                \end{equation}

            Donde $\vec{b}$ es un vector con valores aleatorios binarios del tamaño de las variables de decisión del problema y $\vec{z}$ es el individuo generado por la cruza.

            Constructor:
 
            - No recibe ningún argumento.
  
            Métodos: 
 
            - *__\_\_call\_\_.__* Este método nos permite hacer que nuestra clase se comporte como una función. 
  
            Argumentos:
 
            - `population.` Arreglo bidimensional de *numpy*. Cada fila es un individuo de la población actual.
            - `parent_ind1.` Arreglo de *numpy* que contiene los índices de los individuos seleccionados de `population` que actuarán como padre 1.
            - `parent_ind2.` Arreglo de *numpy* que contiene los índices de los individuos seleccionados de `population` que actuarán como padre 2.
        
            Valor de retorno:
 
            - Arreglo bidimensional de *numpy*. Cada fila es un individuo generado por la cruza.
        !!! Success ""
            **intermediate**. Operador de cruza intermedia. Sean $\vec{x}$ y $\vec{y}$ los padres 1 y 2, respectivamente, el nuevo individuo $\vec{z}$ está dado por:
            \begin{equation}
            z_{i} = \alpha \cdot x_{i} + (1 - \alpha) \cdot y_{i}
            \end{equation}
            
            Donde $\alpha$ es un parámetro proporcionado por el usuario.

            Constructor:
            
            - `Alpha.`  Proporción en la que contribuye cada padre para generar al nuevo individuo. Por defecto es 0.5.
            
            Métodos:
            
            - *__\_\_call\_\_.__* Este método nos permite hacer que nuestra clase se comporte como una función.
            
            Argumentos:
            
            - `population.` Arreglo bidimensional de numpy. Cada fila es un individuo de la población actual.
            - `parent_ind1.` Arreglo de numpy que contiene los índices de los individuos seleccionados de population que actuarán como padre 1.
            - `parent_ind2.` Arreglo de numpy que contiene los índices de los individuos seleccionados de population que actuarán como padre 2.
        
            Valor de retorno:

            - Arreglo bidimensional de *numpy*. Cada fila es un individuo generado por la cruza.

    === "Algoritmos genéticos"

        !!! Success ""
            **n_point_crossover.** Este operador es una generalización de la cruza de un punto. Dados dos padres, se crean dos nuevos individuos. Para ello se seleccionan de manera aleatoria $n$ puntos de cruza. Los nuevos hijos van copiando posición a posición la información de uno de los padres. Cada vez que se encuentra un punto de cruza, intercambian el padre del cual están realizando la copia.
 
            Constructor:
            
            - `n_cross.` Número de puntos de cruza, por defecto es 1.
            
            Métodos:
            
            - *__\_\_call\_\_.__* Este método nos permite hacer que nuestra clase se comporte como una función.
            
            Argumentos:

            - `X.` Arreglo bidimensional de *numpy* que contiene el conjunto de individuos de la población actual (`parent_population_x`). Cada fila es un individuo de la población y cada columna corresponde a una variable de decisión.
            - `parent_ind1.` Índices de los individuos que son seleccionados para actuar como padre 1.
            - `parent_ind2.`Índices de los individuos que son seleccionados para actuar como padre 2.
            
            Valor de retorno:

            - Arreglo bidimensional con la población de nuevos individuos. Cada fila es un individuo de la población y cada columna corresponde a una variable de decisión.
        
        !!! Success ""
            **uniform_crossover.** Dados dos padres $P_1$ y $P_2$, se crean dos nuevos individuos $H_1$ y $H_2$ empleando un cambio entre la información proporcionada por el padre que le corresponderá a cada hijo. La información será seleccionada del padre $P_{i}$ con una probabilidad $p_c$ para el hijo $H_i$. La cruza se realiza de la siguiente manera:
            \begin{equation}
            H_{1,i}, H_{2,i} =
            \begin{cases}
            P_{1,i}, P_{2,i} & si & R_i \le p_c \\
            P_{2,i}, P_{1,i} & si & R_i > p_c \\
            \end{cases}
            \end{equation}

            Donde $R$ es un vector que indica un número aleatorio entre $[0,1]$.

            Constructor:
            
            - `flip_prob.` Probabilidad de que una posición sea considerada como punto de cruza.
            
            Métodos:
            
            - *__\_\_call\_\_.__* Este método nos permite hacer que nuestra clase se comporte como una función.
            
            Argumentos:

            - `X.` Arreglo bidimensional de *numpy* que contiene el conjunto de individuos de la población actual (`parent_population_x`). Cada fila es un individuo de la población y cada columna corresponde a una variable de decisión.
            - `parent_ind1.` Índices de los individuos que son seleccionados para actuar como padre 1.
            - `parent_ind2.`Índices de los individuos que son seleccionados para actuar como padre 2.
        
            Valor de retorno:
            
            - Arreglo bidimensional con la población de nuevos individuos. Cada fila es un individuo de la población y cada columna corresponde a una variable de decisión.

        !!! Success ""
            **permutation_order_crossover.** Operador empleado para permutaciones. Dados dos padres $P_1$ y $P_2$, genera dos nuevos individuos $H_1$ y $H_2$. Para el primer hijo $H_1$, selecciona un segmento aleatorio (longitud variable) del padre $P_1$, este segmento es copiado a $H_1$ en las mismas posiciones. Las posiciones restantes son completadas con la información del padre $P_2$, de izquierda a derecha, sin considerar los elementos que aparecen en el segmento copiado del padre $P_1$. Para el segundo hijo $H_2$, se realiza el mismo procedimiento pero intercambiando a los padres.

            Constructor:
            
            - Ningún parámetro al inicializar.

            Métodos:
            
            - *__\_\_call\_\_.__* Este método nos permite hacer que nuestra clase se comporte como una función.
            
            Argumentos:

            - `X.` Arreglo bidimensional de *numpy* que contiene el conjunto de individuos de la población actual (`parent_population_x`). Cada fila es un individuo de la población y cada columna corresponde a una variable de decisión.
            - `parent_ind1.` Índices de los individuos que son seleccionados para actuar como padre 1.
            - `parent_ind2.`Índices de los individuos que son seleccionados para actuar como padre 2.
            
            Valor de retorno:

            - Arreglo bidimensional con la población de nuevos individuos. Cada fila es un individuo de la población y cada columna corresponde a una variable de decisión.

        !!! Success ""
            **simulated_binary_crossover.** Operador para representación real. Dados dos padres $P_1$ y $P_2$, genera dos nuevos individuos $H_1$ y $H_2$ de la siguiente forma:
            \begin{equation}
                H_1 = 0.5 [(P_1 + P_2) - \beta | P_2 - P_1 |]\\
                H_2 = 0.5 [(P_1 + P_2) + \beta | P_2 - P_1 |]\\
            \end{equation}
            Donde $\beta$ se define como sigue:
            \begin{equation}
            \beta =
            \begin{cases}
            (2u)^{\frac{1}{n_{c}+1}} & si & u \le 0.5,\\
            \left(\frac{1}{2(1-u)}\right)^{\frac{1}{n_{c}+1}} & si & u > 0.5 \\
            \end{cases}
            \end{equation}
            Regularmente, $n_{c}$ es igual con 1 ó 2 y $u \in [0,1]$.

            Constructor:
            
            - `n_c.` Parámetro proporcionado por el usuario, por defecto es 1.
            
            Métodos:
            
            - *__\_\_call\_\_.__* Este método nos permite hacer que nuestra clase se comporte como una función.
            
            Argumentos:

            - `X.` Arreglo bidimensional de *numpy* que contiene el conjunto de individuos de la población actual (`parent_population_x`). Cada fila es un individuo de la población y cada columna corresponde a una variable de decisión.
            - `parent_ind1.` Índices de los individuos que son seleccionados para actuar como padre 1.
            - `parent_ind2.`Índices de los individuos que son seleccionados para actuar como padre 2.
            
            Valor de retorno:

            - Arreglo bidimensional con la población de nuevos individuos. Cada fila es un individuo de la población y cada columna corresponde a una variable de decisión.

        !!! Success ""
            **none\_cross.** Operador que no altera la solución actual.
 
            Constructor:
            
            - Ningún parámetro al inicializar.
            
            Métodos:
            
            - *__\_\_call\_\_.__* Este método nos permite hacer que nuestra clase se comporte como una función.
            
            Argumentos:

            - `X.` Arreglo bidimensional de numpy que representa el conjunto de individuos de la población a mutar, donde, cada fila es un individuo de la población y el número de columnas es el número de variables de decisión.
            - `parent_ind1.` Arreglo de numpy que contiene los índices de los individuos seleccionados de la población para ser los individuos de la matriz $𝑋$.
            - `parent_ind2.` Arreglo de numpy que contiene los índices de los individuos seleccionados de la población para ser los individuos de la matriz $Y$.
        
            Valor de retorno:

            - Arreglo bidimensional con la población de individuos generados por la cruza entre los individuos de la matriz 𝑋 y 𝑌.


## Operadores de mutación
Ubicación de operadores:
```python
import pyristic.utils.operators.mutation as pcMut
```
!!! info "Operadores por metaheurística"

    === "Programación evolutiva"

    === "Estrategias evolutivas"

    === "Algoritmos genéticos"

## Selección de padres
Ubicación de esquemas de selección de padres:
```python
import pyristic.utils.operators.selection as pcSelect
```
!!! info "Selección de padres para algoritmos genéticos"



## Selección de sobrevivientes
Ubicación de esquemas de selección de sobrevivientes:
```python
import pyristic.utils.operators.selection as pcSelect
```
!!! info "Esquemas de selección de sobrevivientes"