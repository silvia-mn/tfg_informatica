#import "@preview/zebraw:0.5.5": *
= Implementación del algoritmo Seq2Seq

En el capítulo anterior se ha descrito en detalle la arquitectura seq2seq, el mecanismo de atención y
los principios teóricos que sustentan estos modelos. Hemos comprendido cómo un codificador transforma
una secuencia de entrada en una representación latente y cómo un decodificador, apoyado en la atención,
genera paso a paso una secuencia de salida mientras selecciona dinámicamente la información más relevante
del contexto.

En este capítulo pasamos de la teoría a la práctica: presentamos la implementación en PyTorch de un modelo
seq2seq con atención. El objetivo no es únicamente mostrar el código, sino desentrañar cada uno de sus
componentes para entender cómo los bloques conceptuales estudiados previamente se convierten en operaciones
programáticas concretas. A lo largo del recorrido, intercalaremos el código con explicaciones extensas,
de modo que se vea claramente la relación entre teoría y práctica.

== Modelo

El código fuente que se comenta en esta sección corresponde al fichero `seq2seq.py`, es un módulo de python
que define la arquitectura del modelo apoyándose en `pytorch`.

Durante la explicación abordamos también según salta la necesidad conceptos propios de `pytorch` que es conveniente conocer para entender el código.


#align(
  center,
  figure(
    ```python
    import random
    import torch
    from torch import nn
    ```,
    caption: [Importaciones necesarias para la construcción del modelo seq2seq con atención.],
  ),
)
El módulo comienza con las importaciones necesarias: la librería estándar `random` para manejar decisiones
estocásticas (como el _teacher forcing_), y `torch` junto con `torch.nn` para construir la red neuronal.
Estas son las bases sobre las que se levantará toda la arquitectura seq2seq con atención.

#align(
  center,
  figure(
    ```python
    def layer_init(layer, w_scale=1.0):
        nn.init.kaiming_uniform_(layer.weight.data)
        layer.weight.data.mul_(w_scale)
        nn.init.constant_(layer.bias.data, 0.0)
        return layer
    ```,
    caption: "Inicialización de capas lineales con Kaiming uniform y ajuste de pesos y sesgos.",
  ),
)

La función layer_init se encarga de inicializar de manera cuidadosa las capas lineales del modelo. Usa
la inicialización de Kaiming, que es adecuada para redes profundas con funciones de activación tipo ReLu,
y asegura que los sesgos comiencen en cero. Este tipo de detalle evita problemas comunes como gradientes
mal escalados desde el inicio del entrenamiento.

La variante Kaiming uniform asigna los pesos a partir de una distribución uniforme dentro de un rango
calculado en función del número de unidades de entrada a la capa. El objetivo es mantener estable la
varianza de las activaciones a través de las distintas capas de la red, evitando que se amplifiquen o
se reduzcan en exceso.

Este método fue propuesto específicamente para trabajar con activaciones ReLU, que anulan los valores
negativos. Si no se aplicara un esquema de inicialización adecuado, la red podría caer en zonas muertas
o generar gradientes extremadamente pequeños, ralentizando o incluso bloqueando el aprendizaje. Al aplicar
Kaiming uniform se logra que la propagación hacia adelante y hacia atrás se mantenga equilibrada desde
el inicio del entrenamiento, favoreciendo una convergencia más rápida y estable.

=== Codificador

#align(
  center,
  figure(
    ```python
    class Encoder(nn.Module):
        def __init__(self, enc_feature_size, hidden_size, num_gru_layers, dropout):
            super().__init__()
            self.gru = nn.GRU(
                enc_feature_size,
                hidden_size,
                num_gru_layers,
                batch_first=True,
                dropout=dropout,
            )
    ```,
    caption: "Definición de la clase Encoder basada en GRU",
  ),
)
El codificador (_encoder_) está basado en una red recurrente del tipo GRU. Recibe una secuencia de entrada con múltiples
características y produce representaciones ocultas que condensan la información temporal. Como se explicó
anteriormente en la teoría, el encoder es la parte encargada de transformar la secuencia original en
un espacio latente que el decodificador podrá explotar.

Más adelante se explorará la posibilidad de usar redes neuronales LSTM, sustituyendo a GRU. De momento
continuamos con GRU.

#align(
  center,
  figure(
    zebraw(
      highlight-lines: (
        (
          1,
          [
            Tensor de entrada:
            - `inputs`: (tamaño de lote, longitud de secuencia de entrada, número de características del codificador)
          ],
        ),
        ..range(2, 4),
        (
          3,
          [
            Tensores de salida:
            - `output`: (tamaño de lote, longitud de secuencia de entrada, tamaño oculto)
            - `hidden`: (número de capas GRU, tamaño de lote, tamaño oculto)
          ],
        ),
      ),
      ```python
      def forward(self, inputs):
          output, hidden = self.gru(inputs)
          return output, hidden
      ```,
    ),
    caption: "Método forward del codificador.",
  ),
)

El método `forward` del codificador devuelve tanto la secuencia completa de estados ocultos como el último
estado de la GRU. El primero servirá para el mecanismo de atención y el segundo inicializará al decodificador.


=== Decodificador Base

#align(
  center,
  figure(
    ```python
    class DecoderBase(nn.Module):
        def __init__(self, device, dec_target_size, target_indices):
            super().__init__()
            self.device = device
            self.target_indices = target_indices
            self.target_size = dec_target_size
    ```,
    caption: "Definición del decodificador base con parámetros generales",
  ),
)

El decodificador base establece la estructura común para todos los decodificadores que se quieran construir
encima (Es una clase pensada para ser extendida por subclases). Define variables clave como el tamaño
de los objetivos, a qué índices de la salida del decodificador de corresponden las variables objetivo
y en qué dispositivo (CPU o GPU) se ejecutará el cálculo.

A lo largo del modulo se incluyen comentarios sobre las dimensiones que tienen los distintos tensores
con los que trabaja ya que tener sus dimensiones presentes en todo momento ayuda a razonar sobre el código
y a entender también el contenido de los tensores.

#align(
  center,
  figure(
    zebraw(
      highlight-lines: (
        (
          1,
          [
            Dimensiones de los tensores de entrada:
            - `inputs`: (tamaño de batch, longitud de secuencia de salida, número de características del decodificador)
            - `hidden`: (número de capas GRU, tamaño de batch, dimensión oculta), es decir, el último estado oculto
            - `enc_outputs`: (tamaño de batch, longitud de secuencia de entrada, tamaño oculto)
          ],
        ),
        (
          10,
          [
            Dimensiones: (tamaño de batch, 1, número de características del decodificador)
          ],
        ),
        ..range(12, 15),
        (
          14,
          [`run_single_recurrent_step` debe ser implementado por subclases],
        ),
        ..range(23, 25),
        (
          24,
          [Si se aplica _teacher forcing_, usar el objetivo de este paso como la siguiente entrada; de lo contrario, usar la predicción],
        ),
      ),
      ```python
       def forward(self, inputs, hidden, enc_outputs, teacher_force_prob=None):
            batch_size, dec_output_seq_length, _ = inputs.shape

            outputs = torch.zeros(
                batch_size,
                dec_output_seq_length,
                self.target_size,
                dtype=torch.float,
            ).to(self.device)
            curr_input = inputs[:, 0:1, :]
            for t in range(dec_output_seq_length):
                dec_output, hidden = self.run_single_recurrent_step(
                    curr_input, hidden, enc_outputs
                )
                outputs[:, t : t + 1, :] = dec_output

                teacher_force = (
                    random.random() < teacher_force_prob
                    if teacher_force_prob is not None
                    else False
                )
                curr_input = inputs[:, t : t + 1, :].clone()
                if not teacher_force:
                    curr_input[:, :, self.target_indices] = dec_output
            return outputs
      ```,
    ),
    caption: "Método forward del decodificador base para el recorrido paso a paso.",
  ),
)

La función forward del decodificador base implementa la lógica de tratar paso a paso la secuencia de
salida. Aquí se decide en cada instante si usar teacher forcing o confiar en la predicción anterior.
Esto refleja la dificultad práctica en el entrenamiento de modelos secuenciales: encontrar un balance
entre guiar al modelo con datos reales y dejarle explorar su propia dinámica.

Como se indica, la lógica propiamente del decodificador tendrá que ser implementada en subclases del decodificador base.

=== Atención

#align(
  center,
  figure(
    zebraw(
      highlight-lines: (
        ..range(4, 6),
        (
          5,
          [
            El tamaño oculto para la salida de `attn` (y entrada de `v`) puede ser cualquier número.
            Además, usar dos capas permite tener una función de activación no lineal entre ellas.
          ],
        ),
      ),
      ```python
      class Attention(nn.Module):
            def __init__(self, hidden_size, num_gru_layers):
                super().__init__()
                self.attn = nn.Linear(2 * hidden_size, hidden_size)
                self.v = nn.Linear(hidden_size, 1, bias=False)
      ```,
    ),
    caption: "Implementación del mecanismo de atención",
  ),
)

El módulo de atención se implementa aquí como una red pequeña que compara el estado oculto actual del
decodificador con cada salida del encoder. La capa lineal `attn` calcula una representación intermedia,
y la proyección `v` la condensa en un valor escalar (como se observa en el tamaño de salida de la capa, 1) que representa la “energía” de atención.

#align(
  center,
  figure(
    zebraw(
      highlight-lines: (
        (
          1,
          [
            Tensores de entrada:
            - `decoder_hidden_final_layer`: (tamaño de lote, tamaño oculto)
            - `encoder_outputs`: (tamaño de lote, longitud de secuencia de entrada, tamaño oculto)
          ],
        ),
        ..range(2, 4),
        (
          3,
          [Repetir el estado oculto del decodificador tantas veces como la longitud de la secuencia de entrada],
        ),
        (
          4,
          [Comparar el estado oculto del decodificador con cada salida del codificador usando una capa _tanh_ entrenable],
        ),
        (
          7,
          [`weightings`: (tamaño de lote, longitud de secuencia de entrada)],
        ),
      ),
      ```python
      def forward(self, decoder_hidden_final_layer, encoder_outputs):
          hidden = decoder_hidden_final_layer.unsqueeze(1).repeat(
              1, encoder_outputs.shape[1], 1)
          energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
          attention = self.v(energy).squeeze(2)
          weightings = torch.nn.functional.softmax(attention, dim=1)
          return weightings
      ```,
    ),
    caption: "Cálculo de pesos de atención mediante softmax",
  ),
)

En el método `forward` de atención se obtiene el peso asignado a cada posición de la secuencia de entrada. La normalización con _softmax_ garantiza que las ponderaciones sumen uno, de forma que pueden interpretarse como una distribución de probabilidad sobre los pasos de la secuencia. No entramos en detalle, ya que es una implementación directa de los fundamentos teóricos explicados en elcapítulo anterior.

=== Decodificador con atención

#align(
  center,
  figure(
    zebraw(
      highlight-lines: (
        (
          19,
          [GRU toma el objetivo del paso anterior y la suma ponderada de los estados ocultos del codificador (resultado de la atención)],
        ),
        (
          27,
          [
            La capa de salida toma la salida del estado oculto del decodificador, la suma ponderada según la atención y la entrada del decodificador.
          ],
        ),
      ),
      ```python
      class DecoderWithAttention(DecoderBase):
        def __init__(
            self,
            dec_feature_size,
            dec_target_size,
            hidden_size,
            num_gru_layers,
            target_indices,
            dropout,
            device,
        ):
            super().__init__(
                device,
                dec_target_size,
                target_indices,
            )
            self.attention_model = Attention(hidden_size, num_gru_layers)
            self.gru = nn.GRU(
                dec_feature_size + hidden_size,
                hidden_size,
                num_gru_layers,
                batch_first=True,
                dropout=dropout,
            )
            self.out = layer_init(
                nn.Linear(
                    hidden_size + hidden_size + dec_feature_size,
                    dec_target_size,
                )
            )
      ```,
    ),
    caption: "Definición del decodificador con atención.",
  ),
)

El decodificador con atención combina las ideas anteriores: recibe la entrada del paso actual, calcula
el contexto mediante la atención y concatena todo para alimentar a la GRU. Posteriormente, una capa lineal
genera los parámetros de salida. Aquí primero se definen las capas necesarias para todo el proceso, que
se encapsulará en el método que se quedó pendiente de implementar (`run_single_recurrent_step`).

#align(
  center,
  figure(
    zebraw(
      highlight-lines: (
        (
          1,
          [
            Tensores de entrada:
            - `inputs`: (tamaño de lote, 1, número de características del decodificador)
            - `hidden`: (número de capas GRU, tamaño de lote, tamaño oculto)
            - `enc_outputs`: (tamaño de lote, longitud de secuencia de entrada, tamaño oculto)
          ],
        ),
        (
          2,
          [
            Obtener los pesos de atención
            - `weightings`: (tamaño de lote, longitud de secuencia de entrada)
          ],
        ),
        (
          3,
          [
            Luego calcular la suma ponderada
            - `weighted_sum`: (tamaño de lote, 1, tamaño oculto)
          ],
        ),
        (
          4,
          [
            Después introducir en la GRU
            - entradas de GRU: (tamaño de lote, 1, número de características del decodificador + tamaño oculto)
            - `output`: (tamaño de lote, 1, tamaño oculto)
          ],
        ),
        (
          5,
          [
            Obtener predicción
            - entrada de `out`: (tamaño de lote, 1, tamaño oculto + tamaño oculto + número de objetivos)
          ],
        ),
        (
          7,
          [
            Tensores de salida:
            - `output`: (tamaño de lote, 1, número de objetivos)
            - `hidden`: (número de capas GRU, tamaño de lote, tamaño oculto)
          ],
        ),
      ),
      ```python
      def run_single_recurrent_step(self, inputs, hidden, enc_outputs):
          weightings = self.attention_model(hidden[-1], enc_outputs)
          weighted_sum = torch.bmm(weightings.unsqueeze(1), enc_outputs)
          output, hidden = self.gru(torch.cat((inputs, weighted_sum), dim=2), hidden)
          output = self.out(torch.cat((output, weighted_sum, inputs), dim=2))
          output = output.reshape(output.shape[0], output.shape[1], self.target_size)
          return output, hidden
      ```,
    ),
    caption: "Ejecución de un paso recurrente del decodificador con atención.",
  ),
)

En cada paso, el decodificador con atención calcula primero las ponderaciones, luego el vector de contexto
como combinación de las salidas del encoder, y lo introduce en la GRU junto con la entrada actual. La
salida se transforma en la predicción final para ese instante. Esta combinación permite que el decodificador
se “enganche” dinámicamente a diferentes partes de la secuencia de entrada según lo requiera cada predicción.

=== Juntándolo todo. Modelo Seq2Seq.

#align(
  center,
  figure(
    ```python
    class Seq2Seq(nn.Module):
        def __init__(self, encoder, decoder, lr, grad_clip):
            super().__init__()
            self.encoder = encoder
            self.decoder = decoder
            self.opt = torch.optim.Adam(self.parameters(), lr)
            self.loss_func = Seq2Seq.compute_smape
            self.grad_clip = grad_clip
    ```,
    caption: "Clase Seq2Seq que unifica encoder, decodificador y entrenamiento.",
  ),
)

Finalmente la calse Seq2Seq unifica todo el sistema. Primero como en el resto de módulos se inicializan los componentes y los parámetros que afectarán al modelo.
Aparece nuevamente el optimizador Adam que ya se ha descrito, que se ajusta con el parametro `lr` (_learning rate_). Además se especifica la función de pérdida SMAPE.

El parámetro `grad_clip` merece mención especial: se utiliza para aplicar _gradient clipping_, una técnica
que limita la magnitud de los gradientes durante el retropropagado. Esto resulta muy útil en modelos
recurrentes, donde los gradientes pueden crecer de forma explosiva y desestabilizar el entrenamiento.
Al imponer un tope, `grad_clip` garantiza que las actualizaciones de los parámetros sean más estables
y que el entrenamiento progrese de manera controlada.

#align(
  center,
  figure(
    ```python
    @staticmethod
    def compute_smape(prediction, target):
        return (
            torch.mean(
                torch.abs(prediction - target)
                / ((torch.abs(target) + torch.abs(prediction)) / 2.0 + 1e-8)
            )
            * 100.0
        )
    ```,
    caption: "Método estático que calcula el SMAPE (Symmetric Mean Absolute Percentage Error) entre predicciones y valores reales",
  ),
)

Aquí vemos la implementación del cálculo de SMAPE como un método estático dentro de la clase. El hecho
de declararlo con `@staticmethod` implica que no depende de la instancia concreta del modelo, sino que
se puede invocar directamente desde la clase sin necesidad de crear un objeto. Dado que la fórmula ya
fue explicada en detalle en un capítulo previo, basta con señalar que este método proporciona una medida
porcentual del error relativo entre la predicción y el valor real.

#align(
  center,
  figure(
    zebraw(
      highlight-lines: (
        (
          1,
          [
            Tensores de entrada:
            - `prediction`: (tamaño de lote, longitud de secuencia del decodificador, número de objetivos)
            - `target`: (tamaño de lote, longitud de secuencia del decodificador, número de objetivos)
          ],
        ),
        (
          3,
          [Devuelve el tensor de pérdida si está en modo entrenamiento, sino el valor escalar],
        ),
      ),
      ```python
      def compute_loss(self, prediction, target):
          loss = self.loss_func(prediction, target)
          return loss if self.training else loss.item()
      ```,
    ),
    caption: "Cálculo de la función de pérdida en el modelo Seq2Seq",
  ),
)

Esta es una función de utilidad que devuelve el resultado del cómputo de la función de pérdida como un tensor cuando el modelo está
en modo de entrenamiento, mientras que en modo de evaluación devuelve un valor
numérico. Esto es útil porque durante el entrenamiento se necesita mantener el grafo de
cómputo para poder hacer _backpropagation_, mientras que en la evaluación basta con obtener un valor (_float_)
para inspección o registro.

Más detalladamente, en PyTorch, cada operación que involucra tensores con cierto parámetro activado
(`requires_grad=True`) se añade a un grafo dinámico de cómputo. Este grafo registra las transformaciones
realizadas sobre los datos para que, en el momento de llamar a `.backward()`, la librería pueda aplicar
automáticamente la regla de la cadena y calcular los gradientes de todos los parámetros implicados. Mientras
el modelo está en modo de entrenamiento, es fundamental conservar dicho grafo junto con la pérdida para
permitir la retropropagación. Sin embargo, cuando el modelo se encuentra en fase de evaluación o inferencia,
no se necesita este mecanismo y resulta más eficiente “desprenderse” del grafo, devolviendo simplemente
un valor numérico (` loss.item()`) en lugar de un tensor que siga conectado a la cadena de operaciones.

#align(
  center,
  figure(
    zebraw(
      highlight-lines: (
        (
          1,
          [
            Tensores de entrada:
            - `enc_inputs`: (tamaño de lote, longitud de secuencia de entrada, número de características del codificador)
            - `dec_inputs`: (tamaño de lote, longitud de secuencia de salida, número de características del decodificador)
          ],
        ),
        (
          2,
          [
            Salidas del codificador:
            - `enc_outputs`: (tamaño de lote, longitud de secuencia de entrada, tamaño oculto)
            - `hidden`: (número de capas GRU, tamaño de lote, dimensión oculta), es decir, el último estado oculto
          ],
        ),
        (
          3,
          [
            Salidas finales:
            - `outputs`: (tamaño de lote, longitud de secuencia de salida, número de objetivos)
          ],
        ),
      ),
      ```python
      def forward(self, enc_inputs, dec_inputs, teacher_force_prob=None):
          enc_outputs, hidden = self.encoder(enc_inputs)
          outputs = self.decoder(dec_inputs, hidden, enc_outputs, teacher_force_prob)
          return outputs
      ```,
    ),
    caption: "Cálculo de la función de pérdida en el modelo Seq2Seq",
  ),
)

Como hemos ido viendo, el método `forward` es el núcleo de cualquier clase `nn.Module` en PyTorch,
ya que define cómo fluyen los datos a través del modelo. Se define en una dirección, hacia delante, y
gracias al grafo de cómputo tenemos el flujo también hacia atrás.

En este caso, se reciben dos secuencias de entrada: `enc_inputs`, correspondiente a los vectores que
entran en el codificador, y `dec_inputs`, que se suministran al decodificador. Primero, los datos atraviesan
la red recurrente del codificador (`self.encoder`), produciendo como salida tanto las representaciones
ocultas de toda la secuencia (`enc_outputs`) como el último estado oculto (`hidden`). Estos elementos
constituyen el contexto que se pasará al decodificador.

A continuación, se invoca al decodificador (`self.decoder`), que utiliza tanto el estado oculto como
las salidas del codificador para generar la secuencia de salida paso a paso. El parámetro opcional `teacher_force_prob`
permite introducir la técnica de _teacher forcing_, que como se explicó en el capítulo teórico, controla
hasta qué punto el modelo usa la salida real de entrenamiento en lugar de su propia predicción durante
el proceso de decodificación. El valor devuelto, `outputs`, representa finalmente las predicciones del
modelo para la secuencia objetivo.

#align(
  center,
  figure(
    zebraw(
      highlight-lines: (
        (
          1,
          [
            Tensores de entrada:
            - `prediction` & `target`: (tamaño de lote, longitud de secuencia, dimensión de salida)
          ],
        ),
        (
          2,
          [Reiniciar gradientes acumulados],
        ),
        (
          4,
          [Calcular gradientes mediante _backpropagation_],
        ),
        ..range(5, 7),
        (
          6,
          [Recorte de gradientes si está configurado para evitar explosión de gradientes],
        ),
        (
          7,
          [Actualizar parámetros del modelo],
        ),
      ),
      ```python
      def optimize(self, prediction, target):
          self.opt.zero_grad()
          loss = self.compute_loss(prediction, target)
          loss.backward()
          if self.grad_clip is not None:
              torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
          self.opt.step()
          return loss.item()
      ```,
    ),
    caption: [Método de optimización que actualiza los parámetros del modelo mediante retropropagación y _grad clipping_],
  ),
)

Este método concentra el ciclo completo de optimización de parámetros en PyTorch. El procedimiento comienza
con self.opt.zero_grad(), que limpia los gradientes acumulados en la iteración anterior; esto es esencial,
ya que PyTorch por defecto acumula los gradientes. Luego se calcula la pérdida invocando a `compute_loss`,
obteniendo un escalar que mide el error actual del modelo (pero, repetimos, dentro de un tensor que guarda el grafo de cómputo).

El paso siguiente es `loss.backward()`, que activa el mecanismo de retropropagación y recorre el grafo
de cómputo dinámico para calcular los gradientes de cada parámetro entrenable. Si se ha especificado
un valor de `grad_clip`, se aplica `torch.nn.utils.clip_grad_norm_`, que evita que los gradientes crezcan
de manera desmesurada y provoquen inestabilidad numérica, un problema común en redes recurrentes. Finalmente,
se llama a `self.opt.step()`, que ejecuta la actualización efectiva de los parámetros según la regla del
optimizador (en este caso, Adam). El valor retornado es la pérdida convertida a número flotante, útil
para su monitorización o registro durante el entrenamiento.

Con esto completamos la parte central de la implementación que es el modelo como un módulo de PyTorch.

== Formato en secuencias

Para que el modelo pueda entrenarse correctamente, es imprescindible preparar los datos en el formato
adecuado. Los pasos de limpieza de datos y _feature engineering_, documentados para el modelo basado
en una red neuronal simple son de utilidad aquí también, sin embargo el formato de entrada para el modelo
Seq2Seq es muy distinto del formato tabular (_dataframe_ en `pandas`) del que partimos.

En esta sección nos centraremos en un módulo de Python, `format_seqs.py`, que implementa justamente las
funciones necesarias para transformar un _dataframe_ de `pandas` en matrices `numpy` con las dimensiones
apropiadas para alimentar el modelo. Dicho módulo se encarga de recorrer los datos, organizarlos por
particiones (en nuestro caso, una partición por paciente), mantener el orden temporal y finalmente devolver
tensores listos para ser convertidos en `torch.Tensor`.

#align(
  center,
  figure(
    ```python
    from copy import copy
    from functools import partial
    import pandas as pd
    import numpy as np
    ```,
    caption: "Importación de librerías necesarias para la manipulación de datos y funciones auxiliares.",
  ),
)

El módulo comienza importando varias librerías estándar. La función `copy` del paquete `copy` se utiliza
para clonar estructuras intermedias sin que las modificaciones posteriores alteren los objetos originales.
El decorador `partial` de `functools` permite fijar ciertos parámetros de una función para luego aplicarla
de manera más cómoda dentro de operaciones como `groupby`. Finalmente, `pandas` y `numpy` son las bibliotecas
fundamentales para manipular series temporales en estructuras tabulares y transformarlas en arrays
multidimensionales, respectivamente.

#align(
  center,
  figure(
    ```python
    def partitioned_series_to_sequences(
        df: pd.DataFrame, seq_len: int, features: list[str], order_key: str
    ):
        result = []
        buf = []
        for i, row in df.reset_index().sort_values(by=order_key).iterrows():
            if i < seq_len:
                buf.append(list(row[features]))
                continue
            result.append(copy(buf))
            buf.pop(0)
            buf.append(list(row[features]))
        if not result:
            return None
        return np.array(result)
    ```,
    caption: "Generación de secuencias deslizantes a partir de una partición ordenada de datos.",
  ),
)

Esta función constituye el núcleo de la preparación de secuencias. Recibe un `DataFrame` que corresponde
a una partición individual (todos los datos de un paciente), junto con una
longitud de secuencia `seq_len`, la lista de características relevantes y una clave de orden temporal.
El procedimiento recorre cada fila en orden, construyendo un búfer (`buf`) de tamaño fijo que se va desplazando
como una ventana móvil sobre los datos. Cuando el búfer alcanza la longitud necesaria, se copia su contenido
en la lista de resultados, y a continuación se elimina el primer elemento para dar paso al siguiente.
De esta manera, cada fila del DataFrame contribuye a la construcción de múltiples secuencias consecutivas
que capturan la dinámica temporal de las variables seleccionadas. Si la partición es demasiado corta
y no se obtiene ninguna secuencia completa, se devuelve un valor nulo.

#align(
  center,
  figure(
    ```python
    def create_seqs(df: pd.DataFrame, partition_key: str, order_key: str, features : list[str], seq_len: int) -> np.array:
        return np.concat(
            list(
                df.groupby(partition_key)
                .apply(
                    partial(
                        partitioned_series_to_sequences,
                        seq_len = seq_len,
                        features=features,
                        order_key=order_key,
                    ),
                    include_groups=False,
                )
                .dropna()
            )
        )
    ```,
    caption: "Construcción de secuencias a partir de todas las particiones del DataFrame.",
  ),
)

Esta segunda función amplía el enfoque anterior. En lugar de procesar únicamente una partición, toma
todo el `DataFrame` y lo divide en grupos definidos por `partition_key` (de manera general podría ser el identificador
de un producto, usuario o sensor, en nuestro caso es la columna con el id del paciente). A cada grupo se le aplica la función `partitioned_series_to_sequences`
mediante un partial, que ya tiene preconfigurados los argumentos `seq_len`, `features` y `order_key`. El resultado
es una colección de secuencias extraídas de cada partición, que luego se concatenan en un único array
de `numpy`. En este paso se unifican las distintas series individuales en un formato común, listo para
ser usado en el entrenamiento del modelo.

#align(
  center,
  figure(
    ```python
    def format_data(
        df: pd.DataFrame, partition_key, order_key,
        encoder_input_features, decoder_input_features,
        output_features, input_seq_length, output_seq_length,
    ) -> tuple[tuple[np.array, np.array, np.array], list[int]]:
        features = list(set(
                encoder_input_features
                + decoder_input_features
                + output_features
            ))
        feature_index_map = {feature: i for i, feature in enumerate(features)}
        encoder_input_features_idx = [
            feature_index_map[feature]
            for feature in encoder_input_features]
        decoder_input_features_idx = [
            feature_index_map[feature]
            for feature in decoder_input_features]
        output_features_idx = [
            feature_index_map[feature]
            for feature in output_features]
        seqs = create_seqs(
            df, partition_key, order_key, features=features,
            seq_len=input_seq_length + output_seq_length,
        )
        encoder_inputs = seqs[
            :, :input_seq_length, encoder_input_features_idx]
        decoder_inputs = seqs[
            :,
            input_seq_length - 1 : input_seq_length + output_seq_length - 1,
            decoder_input_features_idx]
        decoder_outputs = seqs[
            :,
            input_seq_length : input_seq_length + output_seq_length,
            output_features_idx]
        feature_index_map = {
            feature: i
            for i, feature in enumerate(decoder_input_features)
        }
        return (
            (encoder_inputs, decoder_inputs, decoder_outputs),
            [feature_index_map[feature] for feature in output_features],
        )
    ```,
    caption: "Preparación final de entradas y salidas para el modelo Seq2Seq.",
  ),
)

La última función, `format_data`, completa la tarea de convertir los datos crudos en tensores bien estructurados.
En primer lugar, combina todas las características utilizadas por el codificador, el decodificador y la salida
en una única lista, y genera un mapa de índices que asocia cada nombre de característica con su posición
en la matriz. Posteriormente, llama a `create_seqs` para obtener secuencias que incluyen tanto la longitud
de entrada como la de salida. Con esas secuencias construye tres subconjuntos:

- `encoder_inputs`: los valores de entrada que alimentan al codificador, extraídos según las características definidas.

- `decoder_inputs`: los valores de entrada para el decoder, que comienzan un paso antes de la salida y se extienden hasta cubrir toda la longitud de predicción.

- `decoder_outputs`: los valores reales que se quieren predecir, alineados en la ventana temporal correspondiente.

Finalmente, devuelve una tupla con estos tres arrays junto con un mapeo de índices para las características de salida. Con esto, los datos ya están listos para ser transformados en tensores de PyTorch y pasar directamente al entrenamiento del modelo Seq2Seq.

== Otras funciones

Finalmente, antes de pasar a la ejecución del modelo y evaluar los resultados comentamos en esta sección un módulo que juega un papel más complementario pero igualmente imprescindible: `aux.py`.

Este módulo contiene funciones auxiliares para entrenar, evaluar y manejar los datos de un modelo Seq2Seq,
Incluye generadores de lotes, cálculo de probabilidades de _teacher forcing_, selección del mejor modelo
durante el entrenamiento y división de datos en conjunto de entrenamiento y validación. Todas estas utilidades
simplifican la interacción con los tensores de Pytorch y los modelos que hemos definido.

#align(
  center,
  figure(
    ```python
    def batch_generator(data, batch_size):
        encoder_inputs, decoder_inputs, decoder_targets = data
        indices = torch.randperm(encoder_inputs.shape[0])
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i : i + batch_size]
            yield (
                encoder_inputs[batch_indices],
                decoder_inputs[batch_indices],
                decoder_targets[batch_indices],
                None,
            )
    ```,
    caption: "Generador de lotes aleatorios",
  ),
)

#align(
  center,
  figure(
    ```python
    def train(model, data, batch_size, teacher_force_prob):
        model.train()
        epoch_loss = 0.
        num_batches = 0
            for (
              batch_enc_inputs,
              batch_dec_inputs,
              batch_dec_targets,
              _
            ) in batch_generator(data, batch_size):
            output = model(batch_enc_inputs, batch_dec_inputs, teacher_force_prob)
            loss = model.optimize(output, batch_dec_targets)
            epoch_loss += loss
            num_batches += 1
        return epoch_loss / num_batches
    ```,
    caption: [Entrenamiento de un modelo (una _epoch_).],
  ),
)

Este método ejecuta una pasada de entrenamiento sobre todos los lotes, es decir sobre todo el conjunto
de datos, lo que hemos definido como una _epoch_. Primero activa el modo entrenamiento
(`model.train()`) para que PyTorch registre el grafo de cómputo y permita la retropropagación.
Luego recorre los batches generados, calcula la salida del modelo, aplica la optimización y acumula la pérdida.
Al final, devuelve la pérdida promedio por lote.

#align(
  center,
  figure(
    ```python
    def evaluate(model, val_data, batch_size):
        model.eval()
        epoch_loss = 0.
        num_batches = 0
        with torch.no_grad():
            for (
              batch_enc_inputs,
              batch_dec_inputs,
              batch_dec_targets,
              _
            ) in batch_generator(val_data, batch_size):
                output = model(batch_enc_inputs, batch_dec_inputs)
                loss = model.compute_loss(output, batch_dec_targets)
                epoch_loss += loss
                num_batches += 1
        return epoch_loss / num_batches
    ```,
    caption: "Evaluación de un modelo.",
  ),
)

Esta función evalúa el modelo sobre un conjunto de validación. Se activa `model.eval()` para desactivar
la acumulación de gradientes y regularizaciones como `Dropout`. `torch.no_grad()` evita construir el grafo
de cómputo, reduciendo consumo de memoria. Se calcula la pérdida promedio sobre todos los lotes para
monitorear desempeño.

#align(
  center,
  figure(
    ```python
    def calc_teacher_force_prob(decay, indx):
        return decay / (decay + math.exp(indx / decay))
    ```,
    caption: [Cálculo de probabilidad de _teacher forcing_.],
  ),
)
Esta función determina la probabilidad de usar _teacher forcing_ en la época `indx`. La probabilidad decrece
sigún una función *sigmoide inversa* a medida que avanzan las épocas, controlando el equilibrio entre usar las salidas reales
del modelo y los valores correctos como inputs del decoder. Es la función de decaimiento.

El parámetro `decay` controla la velocidad a la que esta probabilidad desciende. De manera quizás un
poco contraintuitiva a mayor valor del parámetro `decay` más lentamente cae la
probabilidad.

#align(center)[
  #figure(
    image("../assets/decay-func.png", width: 85%),
    caption: [Curvas de la función de decaimiento para valores del parámetro `decay` 0.5, 1 y 2.],
  )
]


#align(
  center,
  figure(
    ```python
    def get_best_model(
        model, train_data, val_data,
        batch_size, num_epochs, decay
    ):
        best_val, best_model = float('inf'), None
        for epoch in range(num_epochs):
            start_t = time()
            teacher_force_prob = calc_teacher_force_prob(decay, epoch)
            train_loss = train(
              model, train_data, batch_size, teacher_force_prob
            )
            val_loss = evaluate(model, val_data, batch_size)
            new_best_val = False
            if val_loss < best_val:
                new_best_val = True
                best_val = val_loss
                best_model = deepcopy(model)
            print(
                f'Epoch {epoch+1} => Train loss: {train_loss:.5f},'
                f' Val: {val_loss:.5f},'
                f' Teach: {teacher_force_prob:.2f},'
                f' Took {(time() - start_t):.1f} s'
                f'{"      (NEW BEST)" if new_best_val else ""}'
            )
        return best_model
    ```,
    caption: "Selección del mejor modelo durante entrenamiento.",
  ),
)

Esta función recorre `num_epochs` pasadas de entrenamiento, calculando pérdida de entrenamiento y validación en
cada época. Si la pérdida de *validación* mejora, se hace un `deepcopy` del modelo, conservando la mejor
versión. Esto permite *evitar sobreajuste* y seleccionar automáticamente el modelo más generalizable. La
impresión muestra métricas por _epoch_, incluida la probabilidad de _teacher forcing_.

#align(
  center,
  figure(
    ```python
    def train_val_split(data, p=0.8):
        n = data[0].shape[0]
        indices = random.sample(list(range(n)), k=int(n * p))
        remaining = [i for i in range(n) if i not in indices]
        return (
            tuple(map(lambda arr: torch.Tensor(arr[indices]), data)),
            tuple(map(lambda arr: torch.Tensor(arr[remaining]), data)),
        )
    ```,
    caption: "División de datos en entrenamiento y validación.",
  ),
)
Finalmente tenemos esta función auxiliar que divide un dataset en conjuntos de entrenamiento y validación
según un porcentaje dado `p`. Se generan índices aleatorios para el conjunto de entrenamiento y el resto
se asigna a validación. Cada parte se transforma en tensores de PyTorch, listos para ser usados por los
métodos de entrenamiento y evaluación anteriores.

== Entrenamiento y evaluación

Ya podemos pasar a el entrenamiento y evaluación del modelo. El código restante es sencillo,
tratándose principalmente de la definición de los (hiper-)parámetros. La búsqueda de una configuración
óptima de estos es un problema grande por si mismo. En esta sección, también compararemos varias configuraciones.

El código que comentaremos a continuación se incluye en un cuaderno de jupyter `model.ipynb`.
En este cuaderno importamos los módulos que hemos explicado a lo largo de este capítulo. Como ya hemos mencionado parte de los pasos preliminares
pueden ser reutilizados de lo aplcado para el modelo basado en una red neuronal simple, así que obviaremos esa parte.

#align(
  center,
  figure(
    zebraw(
      highlight-lines: (
        (
          17,
          [`visit_month` es la única covariable],
        ),
        (
          28,
          [80% de los datos para entrenar],
        ),
      ),
      ```python
      from format_seqs import format_data
      from aux import train_val_split
      encoder_input_features = [
          "visit_month",
          "updrs_1",
          "updrs_2",
          "updrs_3",
          "updrs_4",
      ]
      decoder_input_features = [
          "visit_month",
          "updrs_1",
          "updrs_2",
          "updrs_3",
          "updrs_4",
      ]
      output_features = decoder_input_features[1:]
      data, target_indices = format_data(
          scaled_patient,
          partition_key="patient_id",
          order_key="visit_month",
          encoder_input_features=encoder_input_features,
          decoder_input_features=decoder_input_features,
          output_features=output_features,
          input_seq_length=3,
          output_seq_length=3,
      )
      train_data, val_data = train_val_split(data, p = 0.8)
      ```,
    ),
    caption: "Preparación y configuración de datos para entrenamiento del modelo seq2seq con características UPDRS.",
  ),
)

En este fragmento de código se realiza la preparación y configuración de los datos necesarios para entrenar
el modelo seq2seq. Primero se definen las características de entrada tanto para el codificador como para
el decodificador, que incluyen el mes de visita (`visit_month`) como covariable temporal y las cuatro
puntuaciones de la escala UPDRS (`updrs_1` a `updrs_4`) que evalúan diferentes aspectos de los síntomas
del Parkinson. Las características de salida se limitan únicamente a las puntuaciones UPDRS, excluyendo
el mes de visita que actúa como variable de control temporal. Posteriormente, se utiliza la función
`format_data` para transformar los datos en secuencias apropiadas para el modelo, organizando la información
por paciente (`patient_id`) y ordenándola cronológicamente (`visit_month`), con secuencias de entrada
y salida de 3 pasos temporales cada una. Finalmente, se divide el conjunto de datos en entrenamiento
y validación, asignando el 80% de los datos para el entrenamiento del modelo y reservando el 20% restante
para la evaluación de su rendimiento.

Probaremos primero de esta manera, sin incluir las columnas de péptidos y/o proteinas en las características de entrada. Posteriormente las añadiremos y contrastaremos los resultados.

#align(
  center,
  figure(
    ```python
    from seq2seq import Encoder, Seq2Seq, DecoderWithAttention

    enc_feature_size = len(encoder_input_features)
    hidden_size = 32
    num_gru_layers = 1
    dropout = 0.1
    dec_feature_size = len(decoder_input_features)
    dec_target_size = len(output_features)
    device = 'cpu'
    lr = 0.0005
    grad_clip = 1
    batch_size = 100
    num_epochs = 50
    decay = 3 #Lower means faster decay

    encoder = Encoder(enc_feature_size, hidden_size, num_gru_layers, dropout)
    decoder_args = (dec_feature_size, dec_target_size, hidden_size, num_gru_layers, target_indices, dropout, device)
    decoder = DecoderWithAttention(*decoder_args)
    seq2seq = Seq2Seq(encoder, decoder, lr, grad_clip).to(device)
    ```,
    caption: "Definición de hiperparámetros y construcción del modelo Seq2Seq",
  ),
)


En este fragmento se definen los hiperparámetros principales para el entrenamiento del modelo Seq2Seq
con atención, y se construyen las instancias de `Encoder`, `DecoderWithAttention` y `Seq2Seq`. Aquí es donde
se fijan tanto las dimensiones de entrada y salida como la capacidad de la red (tamaño de estado oculto,
número de capas, _dropout_, etc.), además de parámetros de entrenamiento como la tasa de aprendizaje,
`grad_clip` o el número de _epochs_. Muchos de estos parámetros ya han ido aparenciendo pero aprovechamos
este bloque, donde están todos juntos, para repasarlos al completo.

El primer paso es calcular el tamaño de entrada del codificador (`enc_feature_size`) y del decoficador
(`dec_feature_size` y `dec_target_size`) a partir de las listas de características (_features_).

El hiperparámetro `hidden_size` establece la dimensión del vector oculto interno de la GRU, lo que controla
la capacidad de representación del modelo: valores más altos pueden capturar relaciones más complejas,
pero también incrementan el riesgo de sobreajuste.

El parámetro `num_gru_layers` indica cuántas capas GRU se utilizarán tanto en codificador como en decodificador.
Si se aumentara, el modelo tendría más profundidad. `dropout` ayuda a evitar sobreajuste apagando
aleatoriamente un porcentaje de las conexiones durante el entrenamiento.

A nivel de entrenamiento, la tasa de aprendizaje (`lr`) controla la magnitud de las actualizaciones de
los parámetros. `grad_clip` se utiliza para evitar explosiones de gradiente, limitando la norma de los
gradientes en cada paso de optimización.

Otros hiperparámetros definen aspectos prácticos: `batch_size` regula cuántas secuencias se procesan
en paralelo, mientras que `num_epochs` elnumero de pasadas completas durante el entrenamiento al conjunto
total de datos. Finalmente, `decay` controla la velocidad con que disminuye la probabilidad de _teacher
forcing_ (afecta a la funciónde decaimiento).

El codificador se construye pasando el tamaño de entrada, `hidden_size` (tamaño del estado oculto), número de capas GRU y `dropout`. Para
el decodificador se prepara una tupla de argumentos (`decoder_args`) que incluyen tanto dimensiones de
entrada y salida como la lista de índices de salida (`target_indices`), además de `dropout` y el `device` (dispositivo).

Por último, el modelo Seq2Seq se instancia combinando `encoder` y `decoder`, junto con la tasa de aprendizaje
y el valor de gradiente máximo (`grad_clip`). El método `.to(device)` asegura que el modelo se coloque en
el dispositivo adecuado (CPU o GPU).

=== Resultados

Lo único que queda es llamar a la función `get_best_model` con los parámetros adecuados para entrenar al modelo.

==== Modelo sin características de entrada adicionales

Probamos primero con un modelo que haga uso exclusivamente del mes de visita y los valores UPDRS para predecir UPDRS
hasta 18 meses en el futuro. La evaluacion de la función de pérdida (SMAPE) con respecto el conjunto de validación es en este caso aproximadamente un *75%*.

Es conveniente mencionar que la selección del conjunto de validación es aleatoria y por lo tanto podemos obtener distintos resultados ejecutando
el mismo cuaderno varias veces con los mismos parámetros, sin embargo los resultados rondan este valor.

Se trata de una mejora significativa con respecto el modelo de la red neuronal simple, concretamente se presenta una mejora de $30-40%$ con respecto
el modelo de comparación. Hacer uso de una ventana de contexto con valores pasados se demuestra productivo para afinar las predicciones y Seq2Seq nos
ha ofrecido una arquitectura del modelo que ha permitido explotar esa información.

Se incluye también una función para crear gráficos similares a los usados para comprobar los resultados del modelo base, la lógica difiere a la hora
de evaluar el modelo y por ello es una función a parte incluida en el módulo `plot.py`. Presentamos ahora algunos ejemplos del modelo aplicado a los datos
de validación.


#align(center)[
  #figure(
    [
      #image("../assets/seq2seq-model-results-1.png", width: 100%)
      #image("../assets/seq2seq-model-results-2.png", width: 100%)
      #image("../assets/seq2seq-model-results-3.png", width: 100%)
      #image("../assets/seq2seq-model-results-4.png", width: 100%)
    ],
    caption: [Gráficos comparativos de los datos reales con las predicciones del modelo.],
  )
]

La mejora de la métrica SMAPE supone una mejora tangible en la calidad de las predicciones como se puede ver en los ejemplos. El modelo
consigue capturar, si bien de manera imperfecta (todavía estamos hablando de un error de 75%) las tendencias en las distintas categorías.


==== Modelo considerando los datos de proteínas / péptidos

Uno de los objetivos principales a la hora de abordar este proyecto era intentar extraer
información significativa de los conjuntos de datos de péptidos y proteinas mediante el uso de
Seq2Seq, donde otros enfoques habían fallado.

Como vimos para el modelo base, la inclusión de estos datos reducía la precisión del modelo.
En este caso obsevamos que este efecto es mitigado pero concluimos que no es posible,
tampoco de esta manera, extraer valor de estos datos que ayuden a mejorar las predicciones.
El error para ambos casos, incluyendo datos de proteinas e incluyendo datos de péptidos,
vuelve a rondar el *75%*.

==== Modelo con LSTM como red neuronal recurrente alternativa

Se ha implementado también una variante intercambiado las redes neuronales GRU por redes neuronales LSTM. Los cambios respecto a código necesarios para adaptar
el modelo que usa GRU han sido mínimos ya que ambas exponen métodos y constructores muy similares en su implementacción en `pytorch`. El único detalle destacable es
que las redes LSTM tienen como entrada y salida además del tensor principal y el oculto (_hidden_) un tercero llamado _cell state_ que se inicializa a 0, luego hay que
encadenar cuidadosamente estos tensores. Para el cálculo de la atención, reutilizamos el mismo módulo haciendo uso sólo de los tensores _hidden_ al igual que para GRU.

La implementación finalmente resulta en variantes de los componentes de decodificador y codificador, reutilizando el resto de módulos. Se encuentra en el fichero `lstm.py`.

En cuánto a los resultados aquí otra vez obtenemos un error prácticamente equivalente, al rededor de *75%*.

En la siguiente tabla se resumen los desempeños de distintos intentos, cambiando parámetros, carácterísticas de entrada y red neuronal recurrente.


#figure(
  align(center, [
    #table(
      columns: (12%, 10%, 10%, 8%, 10%, 8%, 8%, 8%, 13%, 13%),
      align: center + horizon,
      table.header(
        table.cell([*Entradas*], fill: luma(230), align: center),
        table.cell([*Tipo RNR*], fill: luma(230), align: center),
        table.cell([_*hidden*_], fill: luma(230), align: center),
        table.cell([*Capas*], fill: luma(230), align: center),
        table.cell([_*dropout*_], fill: luma(230), align: center),
        table.cell([*_batch size_*], fill: luma(230), align: center),
        table.cell([*_epochs_*], fill: luma(230), align: center),
        table.cell([*_decay_*], fill: luma(230), align: center),
        table.cell([*Media*], fill: luma(230), align: center),
        table.cell([*Desviación estándar*], fill: luma(230), align: center),
      ),

      [BASE], [GRU], [512], [2], [0.2], [32], [50], [4], [72.05], [1.41],
      [BASE], [GRU], [32], [1], [0.2], [32], [50], [4], [71.52], [1.90],
      [PROTEIN], [GRU], [32], [1], [0.2], [32], [50], [4], [74.34], [2.26],
      [PROTEIN], [GRU], [256], [1], [0.2], [32], [50], [4], [79.34], [4.35],
      [PROTEIN], [GRU], [256], [1], [0.1], [64], [100], [5], [75.28], [0.89],
      [PROTEIN], [LSTM], [256], [1], [0.1], [64], [100], [5], [74.81], [1.43],
      [PROTEIN], [LSTM], [512], [2], [0.1], [64], [100], [5], [75.50], [1.84],
      [PEPTIDE], [LSTM], [512], [2], [0.1], [64], [100], [5], [77.11], [3.46],
      [PEPTIDE], [LSTM], [256], [1], [0.1], [64], [100], [5], [78.33], [1.97],
      [PEPTIDE], [LSTM], [32], [1], [0.1], [64], [100], [5], [77.66], [2.16],
      [PROTEIN], [LSTM], [32], [1], [0.1], [64], [100], [5], [74.93], [0.83],
      [PROTEIN], [LSTM], [64], [1], [0.1], [64], [100], [5], [76.63], [1.74],
      [BASE], [LSTM], [64], [1], [0.1], [64], [100], [5], [73.61], [0.98],
      [BASE], [LSTM], [32], [4], [0.1], [64], [100], [5], [75.98], [1.94],
      [BASE], [LSTM], [4], [1], [0.1], [64], [100], [5], [78.95], [1.95],
      [BASE], [LSTM], [8], [1], [0.1], [64], [100], [5], [76.58], [0.63],
      [BASE], [LSTM], [8], [4], [0.1], [64], [100], [5], [79.31], [1.12],
      [BASE], [LSTM], [16], [4], [0.1], [64], [100], [5], [76.70], [1.67],
      [BASE], [LSTM], [16], [1], [0.1], [64], [100], [5], [75.30], [1.86],
    )
  ]),
  caption: "Resultados de experimentos con distintas configuraciones RNN.",
)

Los datos presentados se han obtenido repitiendo cinco veces cada configuración del modelo, con el fin de reflejar la variabilidad inherente al proceso de entrenamiento. Dicha variación proviene tanto de la forma en que se divide el conjunto en entrenamiento y validación, como de la sensibilidad del ajuste a la tasa de aprendizaje y otros parámetros internos. A partir de estas repeticiones, se calcularon la media y la desviación estándar de las medidas de validación, lo que permite tener una visión más estable del rendimiento sin depender de un único entrenamiento puntual

En los experimentos realizados, las variaciones en el tamaño oculto, número de capas y tasa de dropout no muestran un patrón claro de mejora o empeoramiento sostenido en el rendimiento. Si bien se observan pequeñas fluctuaciones en la media de validación al modificar estos parámetros, las diferencias se mantienen en rangos estrechos y no parecen ser estadísticamente relevantes dadas las desviaciones estándar asociadas.

Al comparar entre los distintos tipos de entrada (BASE, PROTEIN y PEPTIDE), los valores de validación se sitúan en intervalos similares, sin que un conjunto de características se distinga de manera consistente frente a los demás. Esto indica que, al menos con las configuraciones probadas, el tipo de entrada no introduce un efecto determinante en el rendimiento final del modelo.

La comparación entre arquitecturas con GRU y LSTM tampoco refleja una ventaja clara de una sobre otra. En ambos casos los resultados oscilan en márgenes similares y la variabilidad entre repeticiones es comparable. En conjunto, estos resultados sugieren que dentro del rango de hiperparámetros explorado, el modelo no es especialmente sensible a estas variaciones estructurales.

El límite en el rendimiento y la escasa variación frente a los parámetros parecen estar muy ligados al tamaño reducido del conjunto de datos. En primer lugar, al disponer de un volumen limitado de ejemplos, el modelo alcanza rápidamente toda la información disponible, de modo que aumentar el número de capas o el tamaño de las representaciones internas no aporta beneficios adicionales, ya que no hay suficientes patrones nuevos que extraer. En segundo lugar, este mismo tamaño reducido incrementa la proporción de ruido y redundancia en relación con la señal útil: con tan pocos datos, las modificaciones en los hiperparámetros apenas logran explotar información adicional y los resultados permanecen dentro de un rango muy estrecho, sin diferencias significativas.
