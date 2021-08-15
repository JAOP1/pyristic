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
            P_{1,i}, P_{2,i} & \text{Si }  R_i \le p_c \\
            P_{2,i}, P_{1,i} & \text{Si }  R_i > p_c \\
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
            (2u)^{\frac{1}{n_{c}+1}} & \text{Si } u \le 0.5,\\
            \left(\frac{1}{2(1-u)}\right)^{\frac{1}{n_{c}+1}} & \text{Si }  u > 0.5 \\
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

??? example "Modo de uso:"
    ```python
        import pyristic.utils.operators.crossover as pcCross
        import numpy as np 
        #Poblacion.
        population = np.random.randint(0,10,(10,4))
        #Indices de los individuos que seran cruzados
        p1 = np.array([0,3,2,2,6])
        p2 = np.array([1,7,9,8,2])
        #Declaracion de operador con el numero de puntos de cruza.
        cruza_3_puntos = pcCross.n_point_crossover(3) 

        resultado = cruza_3_puntos(population,p1,p2)
        print(resultado)
    ```

## Operadores de mutación
Ubicación de operadores:
```python
import pyristic.utils.operators.mutation as pcMut
```
!!! info "Operadores por metaheurística"

    === "Programación evolutiva"

        !!! Success ""
            **sigma\_mutator.** Operador de mutación en cada una de las soluciones de la población, donde, realiza la mutación de la siguiente manera:

            \begin{equation}
            x'_j = x_j + \sigma'_j \cdot N(0,1)
            \end{equation}
    
            donde $x'_j$ es la variable mutada, $x_j$ la variable a mutar, $\sigma'_j$ el tamaño de paso (previamente mutado) y N(0,1) devuelve un número aleatorio usando una distribución normal con media $0$ y desviación estándar igual con $1$.
 
            Constructor:
            
            - No recibe ningún argumento.
            
            Métodos:
            - *__\_\_call\_\_.__* Este método nos permite hacer que nuestra clase se comporte como una función.
 
            Argumentos:
            
            - `X.` Arreglo bidimensional de *numpy* representando a la población de soluciones de la iteración actual, donde, el número de filas es igual al tamaño de la población y el número de columnas es igual al número de variables que tiene el problema que se está resolviendo.
            
            - `Sigma.` Arreglo bidimensional de *numpy*, donde, cada fila representa los tamaños de paso y cada columna es una de las variables que tiene el problema que se está resolviendo.
 
            Valor de retorno:
            
            - Un arreglo bidimensional de *numpy* del mismo tamaño que el arreglo bidimensional de entrada $X$.
 
        !!! Success ""
            **sigma\_ep\_adaptive\_mutator.** Operador de mutación en los tamaños de desplazamiento de cada uno de los individuos de la población. La mutación se realiza de la siguiente manera:

            \begin{equation}
            \sigma'_j = \sigma_j \cdot ( 1 + \alpha \cdot N(0,1))
            \end{equation}
            
            donde $\sigma'_j$ es la variable mutada, $\sigma_j$ la variable a mutar, $\alpha$ parámetro de entrada por el usuario y N(0,1) devuelve un número aleatorio usando una distribución normal con media $0$ y desviación estándar igual con $1$.
 
            Constructor
            
            - `decision_variables.` Número de variables de decisión del problema.
            - `alpha.` Número que será empleado en la actualización de $\sigma$.

            Métodos:
            
            - *__\_\_call\_\_.__* Este método nos permite hacer que nuestra clase se comporte como una función.
 
            Argumentos:
            
            - `X.` Arreglo bidimensional de *numpy* que representa los tamaños de paso de cada uno de los individuos de la población.
        
            Valor de retorno:
                
            - Arreglo bidimensional de *numpy* con los nuevos valores de tamaño de paso.


    === "Estrategias evolutivas"

        !!! Success ""
            **single_sigma_adaptive_mutator**. Muta el valor del tamaño de paso $\sigma$, utilizado para mutar todas las variables de decisión de un individuo. La mutación se realiza como sigue:

            \begin{equation}
            \sigma' = \sigma \cdot e^{\tau \cdot N(0,1)}
            \end{equation}

            Donde $\tau$ es un parámetro que proporciona el usuario. Sea $n$ el número de variables de decisión del problema, su valor por defecto es:
            
            \begin{equation}
            \tau = \frac{1}{\sqrt{n}}
            \end{equation}
 
            Constructor:
            
            - `decision_variables.` Número de variables de decisión del problema.
            
            Métodos:
            
            - *__length.__*  Función auxiliar de la clase `EvolutionStrategy` que indica cuántos tamaños de paso se utilizan para cada individuo. En este caso cada individuo utiliza un único tamaño de paso.
                
                Argumentos:
                
                - No recibe ningún argumento.
                
                Valor de retorno:
                
                - Número de sigma's empleados para cada individuo de la población (este operador retorna 1).
            
            - *__\_\_call\_\_.__* Este método nos permite hacer que nuestra clase se comporte como una función.
 
            Argumentos:
            
            - `sigma.` Arreglo de *numpy* con $m$ valores $\sigma$. $m$ es el tamaño de la población.
            
            Valor de retorno:
            
            - Arreglo *numpy* con los nuevos valores de $\sigma'$.


        !!! Success ""
            **mult_sigma_adaptive_mutator**. Muta los valores de los tamaños de paso, considerando el uso de un tamaño de paso por variable de decisión. La mutación se realiza de la siguiente forma:

            \begin{equation}
                \sigma'_i = \sigma_i \cdot e ^ {\tau' \cdot N(0,1) + \tau \cdot N_i(0,1)}
            \end{equation}

            Donde $\tau$ es un parámetro que proporciona el usuario. Sea $n$ el número de variables de decisión, los valores por defecto son $\tau' = \frac{1}{\sqrt{2n}}$ y $\tau = \frac{1}{\sqrt{2 \sqrt{n}}}$.

            Constructor:
            
            - `decision_variables.` Número de variables de decisión del problema.
            
            Métodos:
            
            - *__length.__*  Función auxiliar para la clase `EvolutionStrategy` que indica cuántos tamaños de paso debe tener cada individuo. En este caso es un tamaño de paso por cada variable de decisión de cada individuo.

                Argumentos:
                
                - No recibe ningún argumento.
            
                Valor de retorno:
                
                - Número de sigma's empleados para cada individuo de la población (este operador retorna el número de variables de decisión del problema).
                    
            - *__\_\_call\_\_.__* Este método nos permite hacer que nuestra clase se comporte como una función.
 
            Argumentos:
            
            - `sigma.`Arreglo bidimensional de *numpy*. Cada fila contiene los valores $\sigma_i$ de uno de los individuos de la población.
            
            Valor de retorno:
            
            - Arreglo *numpy* con los valores mutados $\sigma'_i$.

    === "Algoritmos genéticos"

        !!! Success ""
            **insertion\_mutator.** Operador empleado para generar permutaciones que selecciona aleatoriamente un elemento de la permutación y una nueva posición. Posteriormente, coloca el elemento en la nueva posición y desplaza el resto de los elementos hacia la derecha. Este proceso se repite $n$ veces por cada individuo. Este operador es conocido como *mutación por desplazamiento* y es una generalización de *mutación por inserción*.

            Constructor:
            
            - `n_elements.` Número de elementos a desplazar, por defecto el número es 1.
            
            Métodos:
            
            - *__\_\_call\_\_.__* Este método nos permite hacer que nuestra clase se comporte como una función.
            
            Argumentos:
            
            - `X.` Arreglo bidimensional de *numpy* que representa el conjunto de individuos de la población a mutar. Cada fila es un individuo de la población y cada columna corresponde con una variable de decisión.
            
            Valor de retorno:
            
            - Arreglo bidimensional de *numpy* con la población mutada. Cada fila es un individuo de la población y cada columna corresponde con una variable de decisión.

        !!! Success ""
            **exchange\_mutator.**  Operador utilizado para permutaciones. Intercambia dos posiciones seleccionadas de manera aleatoria del individuo, las demás posiciones de la permutación permanecen igual.

            Constructor:
            
            - Ningún parámetro al inicializar.

            Métodos:
            
            - *__\_\_call\_\_.__* Este método nos permite hacer que nuestra clase se comporte como una función.
  
            Argumentos:
            
            - `X.` Arreglo bidimensional de *numpy* que representa el conjunto de individuos de la población a mutar. Cada fila es un individuo de la población y cada columna corresponde con una de las variables de decisión.
            
            Valor de retorno:
            
            - Arreglo bidimensional de *numpy* con la población mutada. Cada fila es un individuo de la población y cada columna corresponde con una de las variables de decisión. 
        
        !!! Success ""
            **boundary_mutator.** Operador para representación real conocido como *de límite*. Sean LB y UB los límites inferiores y superiores respectivamente, este operador selecciona una posición aleatoria, $i$, del vector $\vec{x}$ y realiza lo siguiente:
            \begin{equation}
            \vec{x}'_{i} = 
            \begin{cases}
            LB & \text{Si } R \le 0.5 \\
            UB & \text{Si }  R > 0.5 \\
            \end{cases}
            \end{equation}
  
            Constructor:
            
            - `bounds.` Límites de las variables de decisión del problema. Acepta los siguientes formatos:
            
                * Arreglo bidimensional de *numpy*. La primera fila contiene los límites inferiores de cada una de las variables de decisión y la segunda fila los límites superiores.
                * Arreglo de *numpy* con dos valores numéricos. El primero es el límite inferior y el segundo es el límite superior. Estos valores serán los límites para todas las variables de decisión del problema.
  
            Métodos:
            
            - *__\_\_call\_\_.__* Este método nos permite hacer que nuestra clase se comporte como una función.
   
            Argumentos:
            
            - `X.` Arreglo bidimensional de *numpy* que representa el conjunto de individuos de la población a mutar. Cada fila es un individuo de la población y cada columna corresponde con una de las variables de decisión.
            
            Valor de retorno:
            
            - Arreglo bidimensional de *numpy* con la población mutada. Cada fila es un individuo de la población y cada columna corresponde a una de las variables de decisión. 

        !!! Success ""
            **uniform_mutator.** Operador para representación real. Sean LB y UB los límites inferiores y superiores respectivamente, este operador selecciona aleatoriamente una posición $i$ del vector $\vec{x}$ y realiza lo siguiente:
            \begin{equation}
            \vec{x}'_{i}= rnd(LB,UB)
            \end{equation}
            
            Donde, $rnd()$ genera un valor aleatorio utilizando una distribución uniforme.
 
            Constructor:
            
            - `bounds.` Límites de las variables de decisión del problema. Acepta los siguientes formatos:
            
                * Arreglo bidimensional de *numpy*. La primera fila contiene los límites inferiores de cada una de las variables de decisión y la segunda fila los límites superiores.
                * Arreglo de *numpy* con dos valores numéricos. El primero es el límite inferior y el segundo es el límite superior. Estos valores serán los límites para todas las variables de decisión del problema.
            
            Métodos:
            
            - *__\_\_call\_\_.__* Este método nos permite hacer que nuestra clase se comporte como una función.
  
            Argumentos:
                
            - `X.` Arreglo bidimensional de *numpy* que representa el conjunto de individuos de la población a mutar. Cada fila es un individuo de la población y cada columna corresponde con una de las variables de decisión.
            
            Valor de retorno:
            
            - Arreglo bidimensional de *numpy* con la población mutada. Cada fila es un individuo de la población y cada columna corresponde a una de las variables de decisión.  
        !!! Success ""
            **non_uniform_mutator.** Operador para representación real que selecciona aleatoriamente una posición $i$ del vector $\vec{x}$ y realiza lo siguiente.
            
            \begin{equation}
            \vec{x}'_{i} = \vec{x}_{i} + N(0, \sigma)
            \end{equation}

            Donde $N$ genera un valor aleatorio utilizando una distribución normal con media $0$ y desviación estándar $\sigma$.

            Constructor:
            
            - `sigma.` Valor numérico con la desviación estándar que se va a utilizar, por defecto es 1. 
  
            Métodos:
            
            - *__\_\_call\_\_.__* Este método nos permite hacer que nuestra clase se comporte como una función.
   
            Argumentos:
            
            - `X.` Arreglo bidimensional de *numpy* que representa el conjunto de individuos de la población a mutar. Cada fila es un individuo de la población y cada columna corresponde con una de las variables de decisión.
            
            Valor de retorno:
            
            - Arreglo bidimensional de *numpy* con la población mutada. Cada fila es un individuo de la población y cada columna corresponde a una de las variables de decisión.   

        !!! Success ""
            **none\_mutator.** Operador que no altera la solución actual. 
 
            Constructor: 
            
            - Ningún parámetro al inicializar.  

            Métodos:
            
            - *__\_\_call\_\_.__* Este método nos permite hacer que nuestra clase se comporte como una función.
    
            Argumentos:
            
            - `X.` Arreglo bidimensional de *numpy* que representa el conjunto de individuos de la población a mutar, donde, cada fila es un individuo de la población y el número de columnas es el número de variables de decisión.
            
            Valor de retorno:
            
            - Arreglo bidimensional de numpy del mismo tamaño que el arreglo de entrada.     

## Selección de padres
Ubicación de esquemas de selección de padres:
```python
import pyristic.utils.operators.selection as pcSelect
```
!!! info "Selección de padres para algoritmos genéticos"

    !!! Success ""
        **roulette\_sampler.** Operador de selección proporcional que simula el comportamiento de una ruleta. La porción de ruleta asignada a cada individuo depende de su valor de aptitud y la aptitud promedio del resto de los individuos.

        Constructor:
        
        - Ningún parámetro al inicializar.
        
        Métodos:
        
        - *__\_\_call\_\_.__* Este método nos permite hacer que nuestra clase se comporte como una función. 
 
        Argumentos:
        
        - `population_f.` Arreglo de *numpy* con valores numéricos que representan los valores obtenidos al evaluar el individuo en la posición $i$ en la función objetivo.
        
        Valor de retorno:
        
        - Arreglo de *numpy* con valores enteros en el intervalo $[0,n)$, donde $n$ es el número total de individuos en la población actual. Cada posición del arreglo indica el índice del individuo de la población seleccionado para actuar como padre.
    
    !!! Success ""
        **stochastic\_universal\_sampler.** Método de selección proporcional que garantiza que cada individuo actúe como padre al menos $m$ veces, donde $m$ es la parte entera del valor esperado del individuo. La decisión de que un individuo sea seleccionado $m+1$ veces, depende de un valor aleatorio.

        Constructor:
        
        - Ningún parámetro al inicializar.

        Métodos:
        
        - *__\_\_call\_\_.__* Este método nos permite hacer que nuestra clase se comporte como una función.
 
        Argumentos:
        
        - `population_f.` Arreglo de numpy con valores numéricos que representan los valores obtenidos al evaluar el individuo en la posición 𝑖 en la función objetivo.
        
        Valor de retorno:
        
        - Arreglo de *numpy* con valores enteros en el intervalo $[0,n)$, donde $n$ es el número total de individuos en la población actual. Cada posición del arreglo indica el índice del individuo de la población seleccionado para actuar como padre.

    !!! Success ""
        **deterministic\_sampler.** Método de selección proporcional que garantiza que cada individuo actúe como padre al menos $m$ veces, donde $m$ es la parte entera del valor esperado del individuo. Para decidir si un individuo actúa como padre $m+1$ veces, se ordenan a los individuos de acuerdo a la parte decimal de su valor esperado y se van seleccionando a los de mayor valor.

        Constructor:
        
        - Ningún parámetro al inicializar.
        
        Métodos:
        
        - *__\_\_call\_\_.__* Este método nos permite hacer que nuestra clase se comporte como una función.
 
        Argumentos:
        
        - `population_f.` Arreglo de *numpy* con valores numéricos que representan los valores obtenidos al evaluar el individuo en la posición 𝑖 en la función objetivo.
        
        Valor de retorno:
        
        - Arreglo de *numpy* con valores enteros en el intervalo $[0,n)$, donde $n$ es el número total de individuos en la población actual. Cada posición del arreglo indica el índice del individuo de la población seleccionado para actuar como padre.

    !!! Success ""
        **tournament_sampler.** Este operador crea grupos aleatorios de individuos de tamaño $m$. En cada grupo, se selecciona al mejor individuo o al peor individuo de acuerdo a su aptitud. La probabilidad de elegir al mejor individuo es $p$ y la probabilidad de elegir al peor individuo es $1-p$.
 
        Constructor:
        
        - `chunks_.` Tamaño de los grupos, por defecto es 2.
        - `prob_.` Probabilidad $p$ con la que se selecciona al mejor individuo.
        
        Métodos:
        
        - *__\_\_call\_\_.__* Este método nos permite hacer que nuestra clase se comporte como una función.
  
        Argumentos:
        
        - `population_f.`  Arreglo de *numpy* con valores numéricos que representan los valores obtenidos al evaluar el individuo en la posición $𝑖$ en la función objetivo. 
        
        Valor de retorno:
            
        - Arreglo de *numpy* con valores enteros en el intervalo $[0,n)$, donde $n$ es el número total de individuos en la población actual. Cada posición del arreglo indica el índice del individuo de la población seleccionado para actuar como padre.
    

## Selección de sobrevivientes
Ubicación de esquemas de selección de sobrevivientes:
```python
import pyristic.utils.operators.selection as pcSelect
```
!!! info "Esquemas de selección de sobrevivientes"

    !!! Success ""

        **merge_selector.** Esquema $(\mu + \lambda)$, selecciona $\mu$ individuos que son obtenidos al unir la población de hijos y la población actual. Los individuos que permanecerán en la próxima generación son aquellos que tengan un mejor valor de aptitud.

        Constructor:
        
        - No recibe ningún argumento.
        
        Métodos:
        
        - *__\_\_call\_\_.__* Este método nos permite hacer que nuestra clase se comporte como una función.
        
        Argumentos:

        - `parent_f.` Arreglo de numpy de la población almacenada en `parent_population_f`, donde, cada componente representa el valor de la función objetivo por el individuo $i$.
        - `offspring_f.` Arreglo de numpy de la población almacenada en `offspring_population_f`, donde, cada componente representa el valor de la función objetivo por el individuo $i$.
        - `features.` Diccionario que tiene las llaves de la información que se desea mantener. Cada llave contiene un arreglo de dos componentes, donde, la primera es la información de `parent_population` y la segunda componente es la información de `offspring_population`.

        Valor de retorno:
        
        - Diccionario con los individuos seleccionados por dicho esquema. Las llaves de este diccionario serán las mismas llaves recibidas en el parámetro features y adicional otra llave con el nombre `parent_population_f`, sin embargo, ahora sólo contendrá la información de los individuos que pasarán a la próxima generación.

    !!! Success ""
        **replacement_selector.** El esquema $(\mu, \lambda)$, reemplaza la población actual con los $\mu$ mejores hijos de acuerdo a su valor de aptitud.

        Constructor:
        
        - No recibe ningún argumento.
        
        Métodos:
        
        - *__\_\_call\_\_.__* Este método nos permite hacer que nuestra clase se comporte como una función.
 
        Argumentos:
      
        - `parent_f.` Arreglo de numpy de la población almacenada en `parent_population_f`, donde, cada componente representa el valor de la función objetivo por el individuo $i$.
        - `offspring_f.` Arreglo de numpy de la población almacenada en `offspring_population_f`, donde, cada componente representa el valor de la función objetivo por el individuo $i$.
        - `features.` Diccionario que tiene las llaves de la información que se desea mantener. Cada llave contiene un arreglo de dos componentes, donde, la primera es la información de `parent_population` y la segunda componente es la información de `offspring_population`.

        Valor de retorno:
        
        - Diccionario con los individuos seleccionados por dicho esquema. Las llaves de este diccionario serán las mismas llaves recibidas en el parámetro features y adicional otra llave con el nombre `parent_population_f`, sin embargo, ahora sólo contendrá la información de los individuos que pasarán a la próxima generación.
