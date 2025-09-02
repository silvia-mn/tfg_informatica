#import "@preview/fletcher:0.5.8" as fletcher: diagram, edge, node


= Solución propuesta
En esta parte del trabajo vamos a centrarnos ya en concreto en la propuesta de solución que
hemos implementado. Como referencias para el capítulo se han utilizado un artículo titulado
"Long Time Series Deep Forecasting with Multiscale Feature
Extraction and Seq2seq Attention Mechanism" #cite(<wang2022long>), 
una introducción a las RNN publicada por IBM #cite(<ibm_rnn>) y otras  #cite(<brenner2023seq2seq>).

== Redes Neuronales Recurrentes (RNN)
Las redes neuronales recurrentes (RNN, por sus siglas en inglés) constituyen una clase de redes neuronales 
especialmente diseñadas para procesar datos secuenciales. A diferencia de arquitecturas tradicionales, 
las RNN poseen una estructura que incorpora memoria interna, lo que les permite retener información de entradas anteriores para influir en el procesamiento de entradas actuales. Esta característica las hace particularmente efectivas en tareas donde el orden temporal o contextual de los datos es relevante, como la predicción de series temporales, el análisis de texto, la traducción automática o el procesamiento de audio.

A nivel estructural, una característica distintiva de las RNN es que comparten sus parámetros (pesos) a través del tiempo, es decir, los mismos pesos se utilizan en cada paso de la secuencia. Este mecanismo reduce significativamente el número de parámetros necesarios y facilita el modelado de relaciones temporales en los datos. No obstante, las RNN tradicionales presentan una limitación importante conocida como el problema del desvanecimiento del gradiente (vanishing gradient problem).


Durante el entrenamiento de una red neuronal, los errores entre las predicciones y los valores reales se propagan hacia atrás mediante un algoritmo denominado backpropagation con el fin de actualizar los pesos. En el caso de las RNN, se utiliza una variante de este algoritmo llamada Backpropagation Through Time (BPTT), la cual extiende la retropropagación a lo largo de todos los pasos temporales de la secuencia.

Sin embargo, cuando la secuencia de entrada es extensa, los gradientes calculados en los pasos finales del tiempo tienden a multiplicarse repetidamente por valores pequeños (derivadas de funciones de activación como la sigmoide o la tangente hiperbólica). Como consecuencia, estos gradientes pueden decrecer exponencialmente al propagarse hacia pasos más antiguos, lo que hace que los pesos correspondientes apenas se actualicen. Este fenómeno provoca que la red tenga dificultades para aprender dependencias a largo plazo, lo que limita su eficacia en tareas donde la información relevante se encuentra en puntos temporales distantes de la secuencia.

Con el objetivo de superar esta limitación, se han propuesto diversas variantes de las RNN:

- *Long Short-Term Memory (LSTM):* Introducida por Hochreiter y Schmidhuber (1997), esta arquitectura incorpora un mecanismo de memoria explícita mediante celdas y compuertas (entrada, olvido y salida) que controlan de manera dinámica qué información debe conservarse o descartarse. Esto permite preservar gradientes estables a lo largo del tiempo y facilita el aprendizaje de dependencias a largo plazo.

- *Gated Recurrent Units (GRU):* Variante simplificada de las LSTM que utiliza solo dos compuertas (actualización y reinicio). Las GRU mantienen un rendimiento comparable al de las LSTM, pero con una estructura más compacta y eficiente en términos computacionales.

- *Redes recurrentes bidireccionales (BRNN):* Permiten a la red considerar tanto el pasado como el futuro de una secuencia al procesar los datos en ambas direcciones. Esto es útil en tareas donde la comprensión del contexto completo es crucial.

- *Modelos codificador-decodificador (Encoder–Decoder):* Utilizados frecuentemente en tareas de secuencia a secuencia (Seq2Seq), estos modelos consisten en un codificador que transforma la secuencia de entrada en un vector de contexto fijo, y un decodificador que genera la secuencia de salida a partir de dicho vector. Dado que el vector de contexto puede ser una limitación en secuencias largas, es común complementar esta arquitectura con mecanismos de atención (attention), que permiten al decodificador enfocarse dinámicamente en partes específicas de la secuencia original.

En todas estas variantes, las RNN aplican funciones de activación no lineales como sigmoide, tanh y ReLU, que afectan directamente al comportamiento de la red y al flujo del gradiente. En particular, tanh es preferida en muchos casos por centrar sus salidas alrededor de cero, lo que mejora la estabilidad numérica y el aprendizaje.



== Redes Seq2Seq

Las redes neuronales Sequence-to-Sequence (Seq2Seq) son una clase de modelos de aprendizaje profundo diseñados para transformar una secuencia de entrada de 
longitud variable en una secuencia de salida también de longitud variable. Este enfoque fue introducido originalmente por Sutskever et al. (2014) para tareas como
la traducción automática, y desde entonces se ha utilizado ampliamente en procesamiento del lenguaje natural (NLP), síntesis de voz, y predicción de series temporales
ya que esta arquitectura es especialmente útil en problemas donde la relación entre entrada y salida no es uno a uno y donde hay dependencias temporales complejas.



Un modelo Seq2Seq consta principalmente de dos componentes:

- *Codificador (Encoder):* Una red neuronal recurrente (RNN) como LSTM o GRU, que procesa la secuencia de entrada paso a paso y resume su contenido 
    en un vector de estado oculto (estado final del codificador). Este vector intenta capturar toda la información relevante de la entrada.

- *Decodificador (Decoder):* Otra RNN (o LSTM/GRU), que toma el vector de estado del codificador como entrada inicial y genera la secuencia de salida. 
    En modelos más avanzados, como los que incorporan attention, 
    el decodificador también puede acceder a todos los estados intermedios del codificador, lo cual mejora el rendimiento en secuencias largas.


#align(center)[
  #figure(
    diagram(
      node-stroke: 1pt,
      node-fill: white,
      edge("u", [Entrada], "->"),
      node((0, -1), [Codificador], inset: 2em, fill: luma(230), label: <E>),
      edge([Vector de contexto], "->"),
      node((3, -1), [Decodificador], inset: 2em, fill: luma(230), label: <D>),
      edge("u", [Salida], "->"),
    ),
    caption: [Descripcion],
  )]




En el contexto de la predicción de series temporales (Forecasting), el codificador toma una secuencia,
o ventana temporal y el decodificador intenta predecir los pasos siguientes, lo que se conoce como
multi-step / N-step forecasting.

#align(center)[
  #figure(
    image("../assets/seq_1.jpg"),
    caption: "Serie temporal Codificador-Decodificador",
  )
]



Más en detalle,

#align(center)[
  #figure(
    image("../assets/seq_2.png",width: 70%),
    caption: "Paso Codificador-Decodificador",
    
  )
]

=== Mecanismo de Atención en Series Temporales

En ciencias cognitivas, se ha observado que los seres humanos tienden a enfocar 
selectivamente su atención en una parte de la información disponible, ignorando el resto.
El mecanismo de atención en redes neuronales es conceptualmente similar a este proceso de
atención visual humana: se centra en extraer la información más relevante dentro de un 
conjunto de datos amplio. Este mecanismo ha sido ampliamente adoptado en tareas de 
aprendizaje profundo basadas en modelos RNN (Redes Neuronales Recurrentes) y CNN (Redes Neuronales Convolucionales).

Bahdanau et al. introdujeron por primera vez el mecanismo de atención en modelos seq2seq, 
con el objetivo de predecir una secuencia objetivo a partir del aprendizaje de la 
información más relevante entre todos los datos anteriores. 
Posteriormente, Vaswani et al. propusieron el modelo Transformer, 
que elimina la recurrencia y se basa completamente en el mecanismo de atención para 
establecer dependencias globales entre la entrada y la salida. Por su parte, Lai et al. 
aplicaron de forma innovadora la atención básica a los estados ocultos de una GRU 
(Unidad Recurrente con Puertas), aprovechando así la información de cada instante de tiempo. 
Para mejorar este mecanismo, Shih et al. introdujeron un concepto de atención multivariable, 
que selecciona las variables más correlacionadas en lugar de los pasos temporales más 
relevantes.

En esta sección se presentan el mecanismo de atención básica y su aplicación en 
arquitecturas seq2seq.


- *Mecanismo de Atención Básica*

El mecanismo de atención básica busca aprender una combinación ponderada de las entradas 
para determinar qué elementos del pasado deben considerarse más relevantes para una tarea 
de predicción. 

Dado un conjunto de datos de series temporales:
$X = [x_1, x_2, ..., x_t]^T$
El cálculo de la atención sobre cada elemento $x_i$ con respecto al elemento actual $x_t$ 
se realiza con las siguientes fórmulas:

$ α_i = f(x_i, x_t)",  para" i = 1, 2, ..., t $ 

$ A = "softmax"([α_1, α_2, ..., α_t]^T) $

$ y = A^T · X $ 

donde $f(·)$ es una función de medida de correlación y la función softmax se define
con respecto al input $α = [α_1, α_2, ..., α_t]^T$ como:

$ A_i = exp(α_i) / sum_{j=1}^{t} exp(α_j) $

$ A = [A_1, A_2, ..., A_t]^T $

Es decir, normaliza el vector en una distribución de probabilidad que consta de $T$ probabilidades 
proporcionales a los exponenciales de los números de entrada.


- *Mecanismo de Atención en Modelos Seq2Seq*

El mecanismo de atención aplicado a modelos secuencia a secuencia (seq2seq) 
permite mejorar la calidad de las predicciones en tareas temporales al considerar 
la relevancia de cada estado oculto codificado.

Dada la serie de entrada: $X = [x_1, x_2, ..., x_t]^T$ Se inicializa el estado 
oculto de la RNN como $h_0$. En la fase del codificador, se actualizan los estados ocultos
secuencialmente como sigue:

$ h_i = "RNN"(x_i, h_{i-1})   "para" i = 1, 2, ..., t $
Y calculamos estos estados de manera recurrente hasta llegar a $i = t$
para obtener como output del codificador el vector $h = [h_1, h_2, ..., h_t]^T$.

Posteriormente, En la fase del decodificador, se aplica atención entre el último estado oculto del 
codificador $h_t$ y todos los estados $h$ para obtener el vector de contexto:

$ s_0 = "Attention"(h_t, h) $

A partir de aquí, se generan predicciones paso a paso. Se inicializa la entrada del 
decodificador como $y_0 = x_t$ y para cada paso $k$, se concatenan $s_{k-1}$ y $y_{k-1}$
y se procesan mediante la RNN:

$ h_{k+t} = "RNN"([s_{k-1}, y_{k-1}], h_{k+t-1}) $

Y después, se aplica nuevamente atención entre $h_{k+t}$ y la secuencia de estados ocultos:

$ s_k = "Attention"(h_{k+t}, h) $

Este mecanismo permite que el modelo aprenda de manera efectiva qué partes del historico de 
de la serie son más relevantes en cada paso del proceso de predicción.


=== Teacher Forcing y Scheduled Sampling
El teacher forcing es un esquema de entrenamiento clásico en modelos de secuencia basados en 
redes neuronales recurrentes (RNNs). Su funcionamiento consiste en que, durante el entrenamiento, 
en cada paso de la secuencia el modelo recibe como entrada el token real previo proveniente de los datos, 
en lugar de la predicción generada por el propio modelo.

Esta estrategia acelera la convergencia y estabiliza el aprendizaje, ya que evita que los errores se 
propaguen en etapas tempranas de la generación. 
Sin embargo, introduce una discrepancia importante entre las fases de entrenamiento e inferencia. 
En la práctica, durante la inferencia los tokens reales no están disponibles y deben ser reemplazados por 
las predicciones del modelo. Esta diferencia, conocida como exposure bias, puede dar lugar a que errores 
iniciales se amplifiquen y acumulen a lo largo de toda la secuencia generada.


Con el objetivo de reducir el sesgo de exposición introducido por el teacher forcing, Bengio et al.#cite(<bengio2015scheduled>)
propusieron la técnica de scheduled sampling, enmarcada dentro de los métodos de curriculum learning. 
La idea central es introducir de manera progresiva el uso de los tokens generados por el propio modelo durante 
el entrenamiento.


En la práctica, para cada paso $t$ de la secuencia, se define una variable binaria 
$z_t tilde "Bernoulli"(epsilon_t)$ donde $epsilon_t in [0,1]$ representa la probabilidad de usar el token real. Así, la entrada al modelo en el instante 
$t$ se define como:

$ tilde(y)_(t_1) = cases(
  y_(t-1)", si" z_t = 1,
  hat(y)_(t-1)", si" z_t = 0,
) $

El comportamiento de $epsilon_t$ a lo largo del entrenamiento está gobernado por una
_decay function_ (función de decaimiento), que controla la transición desde un entrenamiento
completamente guiado $(epsilon_t approx 1)$ hacia un entrenamiento menos guiado $(epsilon_t arrow 0)$.

Entre las funciones de decaimiento más comunes se incluyen:

- *Lineal:* $epsilon_t = "max"(epsilon_0 - k t, 0)$
- *Exponencial:* $epsilon_t = epsilon_0 dot alpha^t ", " alpha < 1$
- *Sigmoide inversa:* $epsilon_t = k / (k + "exp"(t"/"k))$



#align(center)[
  #figure(
    image("../assets/decay_functions.png", width: 50%),
    caption: [Funciones de decaimiento],
  )
]