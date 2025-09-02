= Introducción

La presente memoria aborda un desafío en la intersección entre la tecnología y la medicina: la predicción del avance de la enfermedad de Parkinson mediante técnicas de Machine Learning. Este proyecto nace de la necesidad de desarrollar herramientas más precisas y explicables que permitan mejorar los tratamientos y la gestión de esta enfermedad neurodegenerativa. En las siguientes secciones, se describen los antecedentes, objetivos y metodología que guían este trabajo.

== Motivación

La enfermedad de Parkinson (EP) es una condición neurodegenerativa progresiva que afecta
predominantemente a adultos mayores, aunque también puede presentarse en etapas más tempranas.
Se estima que más de 10 millones de personas en el mundo viven con esta enfermedad, y su
prevalencia está en aumento debido al envejecimiento global de la población. Los síntomas principales
incluyen temblores, rigidez muscular, y dificultad para mantener el equilibrio, pero la enfermedad
también afecta aspectos no motores, como el sueño, el estado de ánimo y la cognición, lo que agrava
significativamente la calidad de vida de los pacientes y sus cuidadores.

La capacidad de predecir la progresión de la EP es crucial tanto para los pacientes como para los
médicos tratantes. Al anticipar la evolución de los síntomas, se pueden diseñar planes de tratamiento
más efectivos, prevenir complicaciones, y optimizar los recursos del sistema de salud. Sin embargo, la
progresión de la enfermedad varía ampliamente entre los pacientes, influenciada por factores como la
edad, el género, las comorbilidades y el acceso a tratamientos. Esta variabilidad presenta un desafío
significativo para los modelos actuales de predicción, que a menudo no logran capturar la complejidad
de la enfermedad.

Las herramientas actuales basadas en análisis estadísticos o modelos de aprendizaje automático
tradicionales, como ARIMA y Support Vector Machines (SVM), han demostrado cierto éxito en tareas
relacionadas, pero enfrentan limitaciones importantes. Estos métodos suelen requerir supuestos
rígidos sobre la naturaleza de los datos y carecen de capacidad para modelar relaciones no lineales
complejas, características de la progresión de la EP. Además, su aplicabilidad se ve reducida en casos
de datos faltantes o ruidosos, una situación común en registros médicos.

Los avances recientes en técnicas de aprendizaje profundo, como los modelos Seq2Seq
(Sequence-to-Sequence, es decir secuencia a secuencia), abren nuevas posibilidades en este ámbito.
Estas arquitecturas, originalmente diseñadas para tareas de traducción automática, han mostrado un potencial
prometedor para analizar series temporales clínicas. Al aprovechar su capacidad para capturar dependencias
temporales de largo alcance y manejar datos heterogéneos, los modelos Seq2Seq podrían ofrecer predicciones más
precisas y robustas. Este trabajo se centra en evaluar el uso de esta tecnología emergente en un contexto
crítico como el de la progresión de la EP, contribuyendo al avance del estado del arte en la intersección
de la salud y el aprendizaje automático.

== Desafío en Kaggle

Un factor determinante que motivó el desarrollo de este proyecto fue el desafío de Kaggle "AMP
Parkinson's Disease Progression Prediction" #cite(<amp_pd_kaggle_2023>). Este reto propone utilizar datos clínicos
y herramientas de aprendizaje automático para predecir la progresión de la enfermedad en pacientes
de forma precisa y reproducible.

=== La plataforma de Kaggle


Kaggle es una plataforma en línea especializada en ciencia de datos y aprendizaje automático, que
funciona como un espacio colaborativo para la resolución de problemas mediante el uso de
modelos predictivos y técnicas estadísticas. Fundada en 2010 y adquirida por Google en 2017, Kaggle
alberga competiciones en las que empresas, instituciones o investigadores proponen retos
basados en conjuntos de datos reales o sintéticos. La comunidad participante, compuesta por
científicos de datos, ingenieros y estudiantes de todo el mundo, diseña modelos que compiten por
mejorar una métrica de evaluación determinada.

#align(center)[
  #figure(
    image("../assets/kaggle-ar21.svg", alt: "Logo de Kaggle"),
    caption: "Logo de Kaggle",
  )
]

Más allá de las competiciones, Kaggle ofrece recursos educativos, notebooks interactivos,
conjuntos de datos públicos y foros activos que fomentan el aprendizaje continuo y el intercambio
de conocimiento. Gracias a su entorno de ejecución basado en la nube, los participantes pueden
entrenar modelos directamente en la plataforma sin necesidad de configuración local, lo que
favorece la inclusión y la accesibilidad. Kaggle se ha consolidado así como una referencia clave
en el ámbito del aprendizaje automático aplicado, especialmente en contextos de evaluación
comparativa y desarrollo de prototipos.

=== _AMP Parkinson's Disease Progression Prediction_

El reto "AMP Parkinson's Disease Progression Prediction" fue organizado en Kaggle, durante 2023, por la iniciativa
Accelerating Medicines Partnership® Parkinson’s Disease (AMP®PD), una colaboración
público-privada liderada por la Foundation for the National Institutes of Health (FNIH) de Estados Unidos,
junto con diversas entidades de la industria biofarmacéutica y centros académicos.

El objetivo principal del reto era desarrollar modelos capaces de predecir la progresión de la enfermedad
de Parkinson en base a datos longitudinales de pacientes. En concreto, se pedía estimar la evolución de
las puntuaciones en la escala MDS-UPDRS (Movement Disorder Society-Sponsored Revision of the
Unified Parkinson’s Disease Rating Scale), una medida clínica estandarizada que evalúa la gravedad de
los síntomas motores y no motores de la enfermedad.

Los datos proporcionados incluían medidas proteómicas y peptídicas obtenidas mediante
espectrometría de masa en líquido cefalorraquídeo (CSF), junto con otra información clínica.
Las observaciones estaban organizadas en varias visitas generalmente cada 6 meses lo que permitía
la modelización de la progresión como un problema de serie temporal multivariable.

Para el desarrollo de este trabajo no se han seguido de forma estricta los requisitos particulares
establecidos en la competición original, dado que esta ya había finalizado en el momento de realizar el
proyecto. Sin embargo, sí se ha mantenido como referencia el objetivo general del reto.

=== Soluciones propuestas (Estado del arte)

Las soluciones con mejor evaluación combinaban enfoque del aprendizaje automático clásico, utilizando, práticamente
de manera exclusiva, metadatos como mes de visita, horizonte de predicción, frecuencia de visitas y frecuencia de
análisis clínicos.

Estos metadatos, que reflejan la evaluación y el seguimiento realizados por profesionales médicos, mejoraron la
precisión del modelo. Sin embargo, resultan insatisfactorio desde el punto de vista clínico, ya que no modelan
la progresión biológica de la enfermedad, sino aspectos relacionados con la atención sanitaria y la frecuencia
de controles.

Los organizadores reconocieron esta limitación y promovieron una segunda fase del estudio centrada en datos
más objetivos, como perfiles proteómicos y peptídicos, buscando establecer relaciones más directas con la
fisiopatología del Parkinson.

Cabe destacar que ninguna de las soluciones ganadoras aplicó modelos Seq2Seq ni arquitecturas específicas para
series temporales complejas, centrándose en modelos tabulares e híbridos. Esto deja un espacio claro para
explorar enfoques que aprovechen mejor la naturaleza longitudinal de los datos clínico-proteómicos.


== Objetivos

El objetivo general de este trabajo es evaluar el potencial de los modelos Seq2Seq en la predicción
del avance de la enfermedad de Parkinson a partir de datos clínicos secuenciales. La hipótesis de
partida es que estas arquitecturas, debido a su capacidad para modelar dependencias temporales
complejas, podrían superar a los enfoques clásicos tanto en precisión como en adaptabilidad.

Este objetivo se alinea con la necesidad creciente de desarrollar sistemas predictivos más robustos,
que sean útiles no sólo desde un punto de vista técnico, sino también clínico. La progresión del
Parkinson es altamente variable entre pacientes, y un modelo capaz de anticipar esta evolución de
forma individualizada podría tener un impacto significativo en la toma de decisiones médicas, el
diseño de tratamientos personalizados y la mejora general del seguimiento del paciente.

Para lograr este objetivo general, se plantean los siguientes objetivos específicos:

- Comprender y analizar en profundidad el conjunto de datos proporcionado en el reto de Kaggle,
  identificando las variables clínicas más relevantes para el pronóstico.

- Diseñar un flujo de preprocesamiento que permita transformar los datos en secuencias
  temporales adecuadas, aplicando técnicas como normalización, imputación de valores faltantes, y
  estructuración en ventanas temporales (necesario para el modelo elegido).

- Implementar y entrenar diferentes configuraciones de modelos Seq2Seq, incluyendo variantes con
  mecanismos de atención (_attention_), distintas capas recurrentes (LSTM, GRU) y otros hiperparámetros.
// Creo que puede ser sencillo hacerlo

- Comparar el rendimiento de los modelos Seq2Seq con una red neuronal de referencia más simple,
  empleando los mismos datos de entrada y misma evaluación, para analizar mejoras reales atribuibles
  al diseño Seq2Seq.

- Evaluar los resultados utilizando la métrica SMAPE (Symmetric Mean Absolute Percentage Error),
  así como el valor de esta función de pérdida sobre un conjunto de datos, de validación, separado
  para este fin. (_validation loss_).

// - Explorar aspectos de interpretabilidad y explicabilidad, incorporando técnicas que permitan
// entender cómo y por qué el modelo toma ciertas decisiones, con vistas a una posible adopción en
// contextos clínicos.

- Documentar los hallazgos obtenidos y elaborar recomendaciones sobre futuras líneas de trabajo,
  incluyendo la integración de más datos (particularmente datos demográficos), el uso de técnicas de _transfer
  learning,_ o la mejora de la transparencia del modelo.

== Herramientas utilizadas

Para el desarrollo del presente trabajo se ha utilizado un conjunto de herramientas tecnológicas
que permiten tanto el procesamiento de datos como el entrenamiento y evaluación de modelos
de aprendizaje automático. A continuación se describen las principales herramientas y entornos
empleados, agrupados por su funcionalidad principal. La selección de estas herramientas está
respaldada por estudios recientes sobre el stack de tecnología estándar en ciencia de datos y
Machine Learning, como #cite(<raschka2020machine>, form: "prose"), que identifica Python y sus principales
bibliotecas como el ecosistema dominante, tanto en investigación como en producción.

=== Lenguaje: Python

Python ha sido el lenguaje de programación principal del proyecto, elegido por su versatilidad,
legibilidad y amplia adopción en la comunidad científica. Permite implementar tanto el flujo de
preprocesamiento como los modelos de aprendizaje profundo. Dentro del ecosistema de Python se
han utilizado las siguientes bibliotecas:

- *NumPy y Pandas*: Bibliotecas fundamentales para la manipulación y análisis de datos
  estructurados. Se han utilizado intensamente para explorar los datos, gestionar valores faltantes,
  transformar variables y construir secuencias temporales.

- *PyTorch*: Framework de desarrollo de redes neuronales utilizado para la implementación de
  los modelos Seq2Seq y la red neuronal de referencia. Se ha preferido frente a otras alternativas
  como TensorFlow por su flexibilidad en la construcción de arquitecturas personalizadas y su
  integración con herramientas de depuración y visualización. Pytorch permite además la aceleración por
  GPU, a pesar de que en este caso en concreto dada la cantidad limitada de datos no ha sido relevante, de cara
  a escalar esta solución reduciría los tiempos de entrenamiento incluso con grandes volúmenes.

- *scikit-learn*: Biblioteca de referencia para tareas de aprendizaje automático tradicional y
  evaluación de modelos. Se ha utilizado para tareas complementarias como la división del conjunto
  de datos en entrenamiento y validación, así como para cálculos estadísticos de referencia.

- *Matplotlib y Seaborn*: Herramientas de visualización utilizadas para representar gráficamente
  tanto las características de los datos como los resultados obtenidos en el proceso de modelado.

=== Entorno: Jupyter Notebooks

Jupyter Notebooks ha sido el entorno de desarrollo interactivo empleado para el análisis
exploratorio, diseño y prueba de modelos. Su formato permite una documentación clara y
estructurada del proceso de trabajo, facilitando tanto la reproducibilidad como la presentación de
resultados. Esta herramienta ha sido recomendada como estándar en entornos de investigación
por su integración con Python y librerías científicas.

=== Composición: Typst

Typst ha sido el sistema de composición tipográfica empleado para la redacción de la memoria. Su
sintaxis clara y capacidad para integrar fácilmente referencias bibliográficas, ecuaciones y formato
técnico lo convierten en una alternativa moderna y eficaz frente a otras herramientas como LaTeX.
Destaca especialmente por su velocidad de compilación, lo que permite iterar de forma mucho más
ágil durante la redacción y maquetación del documento, una ventaja clave en procesos iterativos
como la elaboración de documentos técnicos complejos.

Estas herramientas han sido seleccionadas no sólo por su idoneidad técnica, sino también por su
compatibilidad entre sí y por contar con una amplia comunidad de soporte, lo que ha facilitado su
integración a lo largo del desarrollo del proyecto.

