#import "../uva-template/template.typ": emph-box

= Planificación

La ejecución exitosa de un proyecto de investigación en Machine Learning requiere una planificación cuidadosa
que abarque tanto los aspectos metodológicos como los recursos necesarios. Este capítulo detalla la estrategia
seguida para el desarrollo del presente trabajo, desde la metodología de trabajo adoptada hasta la gestión
temporal y económica del proyecto. La planificación inicial se fundamenta en las mejores prácticas establecidas
en proyectos de ciencia de datos, siguiendo un enfoque iterativo que permite adaptarse a los hallazgos
y desafíos que surgen durante el proceso de investigación. Asimismo, se incluye un análisis retrospectivo
que contrasta la planificación inicial con la realidad del desarrollo, identificando las desviaciones
más significativas y las lecciones aprendidas que pueden ser de utilidad para futuros proyectos similares
en el ámbito del aprendizaje automático aplicado a la medicina.

== Metodología de trabajo

Para el desarrollo de este proyecto se ha adoptado la metodología *CRISP-DM* (_Cross-Industry Standard Process
for Data Mining_), considerada el estándar de facto para proyectos de minería de datos #cite(<wirth2000crisp>),
y que sigue encontrando utilidad en el paradigma actual enfocado a ML #cite(<martinez2019crisp>).

Esta metodología, desarrollada específicamente para proyectos
de ciencia de datos, resulta especialmente adecuada para el contexto de este trabajo por su naturaleza
iterativa y su capacidad para adaptarse a los desafíos propios del análisis de datos clínicos y la construcción
de modelos predictivos.

#align(center)[
  #figure(
    image("../assets/CRISP-DM_Process_Diagram.png", width: 50%),
    caption: [
      Diagrama de la relación entre las distintas fases en CRISP-DM. Se observa que es una metodología fluida
      e iterativa, que permite revisitar fases previas ante nueva información. Se observa además el papel central
      del dato. #cite(<wikimedia_crisp_dm>)],
  )
]

CRISP-DM estructura el proceso en seis fases interconectadas que permiten un desarrollo sistemático y
reproducible:

1. _*Business Understanding*_ (Comprensión del negocio):

  Esta fase inicial tiene como objetivo comprender los objetivos y requerimientos del proyecto desde una
  perspectiva de negocio, para luego convertir este conocimiento en una definición del problema de minería
  de datos y un plan preliminar diseñado para lograr los objetivos.

  En el contexto médico, esto implica entender las necesidades clínicas reales y su impacto
  en la atención al paciente. En este proyecto específico, se definió el problema médico a resolver estableciendo
  los objetivos clínicos de la predicción del avance del Parkinson y su traducción a objetivos de Machine
  Learning medibles. Se analizó el contexto del desafío de Kaggle, identificando las limitaciones de las
  soluciones previas basadas en metadatos clínicos, y se establecieron los criterios de éxito del proyecto:

  #list(
    [Desde una perspectiva técnica (métricas de evaluación).
      El modelo implementado deberá superar a los enfoques empleados en soluciones previas, para ello se medirá
      su desempeño frente a un modelo de referencia (o _baseline_).],

    [Desde una perspetiva clínica (relevancia de las predicciones).
      El modelo deberá utilizar los datos de proteinas y péptidos para mejorar la relevancia con respecto a
      las soluciones previas.],
  )

2. _*Data Understanding*_ (Comprensión de los datos)

  En esta fase se recolecta el conjunto de datos inicial y se procede a familiarizarse con los datos, identificar
  problemas de calidad, descubrir primeras percepciones sobre los datos o detectar subconjuntos interesantes
  para formar hipótesis de información oculta. La comprensión profunda de los datos es fundamental para
  el éxito de cualquier proyecto de minería de datos, ya que determina tanto las técnicas aplicables como
  las limitaciones del análisis posterior. En este trabajo se realizó una exploración inicial exhaustiva
  del conjunto de datos proporcionado, incluyendo un análisis estadístico descriptivo, el análisis de la
  distribución de variables y la detección de valores faltantes.

3. _*Data Preparation*_ (Preparación de los datos):

  Esta fase cubre todas las actividades necesarias para construir el conjunto de datos final que será alimentado
  a las herramientas de modelado. Las tareas de preparación de datos son propensas a ser realizadas múltiples
  veces y no en un orden prescrito, incluyendo la selección de tablas, registros y atributos, así como
  la transformación y limpieza de datos para las herramientas de modelado. Frecuentemente, esta es la fase
  más intensiva en tiempo del proyecto de minería de datos. En este proyecto, esta fase abarcó la limpieza
  y transformación de los datos clínicos y proteómicos, implementando técnicas de imputación de valores
  faltantes adaptadas a la naturaleza temporal de los datos, normalización de variables para garantizar
  la convergencia de los modelos de deep learning, construcción de secuencias temporales estructuradas
  para alimentar los modelos Seq2Seq, y división estratificada del conjunto de datos en entrenamiento y
  validación, manteniendo la coherencia temporal y la representatividad de los diferentes perfiles de progresión
  de la enfermedad.

4. _*Modeling*_ (Modelado):

  En esta fase se seleccionan y aplican varias técnicas de modelado, calibrando sus parámetros a valores
  óptimos. Típicamente, hay varias técnicas para el mismo tipo de problema de minería de datos, y algunas
  técnicas tienen requerimientos específicos sobre la forma de los datos. Por lo tanto, es común regresar
  a la fase de preparación de datos durante esta etapa. La experimentación iterativa es clave en esta fas
  , ya que diferentes algoritmos pueden revelar distintos aspectos de los datos. En este trabajo se implementaron
  y entrenaron los modelos Seq2Seq propuestos, incluyendo variantes con diferentes tipos de capas recurrentes
  (LSTM, GRU) y mecanismos de atención, junto con la red neuronal de referencia para establecer una línea
  base de comparación. Se experimentó con diferentes arquitecturas, hiperparámetros (learning rate, batch
  size, número de capas) y técnicas de regularización, documentando el proceso de entrenamiento y los resultados
  obtenidos en cada iteración.

5. _*Evaluation*_ (Evaluación):

  En esta etapa del proyecto se ha construido un modelo que parece tener alta
  calidad desde una perspectiva de análisis de datos. Sin embargo, es importante evaluar a fondo el modelo
  y revisar los pasos ejecutados para construirlo, para asegurar que el modelo logre apropiadamente los
  objetivos del negocio. La evaluación debe considerar tanto métricas técnicas como criterios de aplicabilidad
  práctica. En este proyecto se evaluó el rendimiento de los modelos utilizando la métrica SMAPE (Symmetric
  Mean Absolute Percentage Error) como medida principal, complementada con el análisis del _validation loss_
  durante el entrenamiento. Se realizaron comparaciones sistemáticas entre los modelos Seq2Seq y la red
  neuronal de referencia, analizando no solo la precisión predictiva sino también la estabilidad del entrenamiento,
  la capacidad de generalización y la robustez frente a diferentes configuraciones de datos de entrada.

6. _*Deployment*_ (Despliegue):

  La creación del modelo no es generalmente el final del proyecto. Aún si el propósito del modelo es incrementar
  el conocimiento de los datos, el conocimiento ganado necesita ser organizado y presentado de una manera
  que el cliente pueda usar. Esta fase se centra en la operacionalización del modelo y la transferencia
  de conocimiento.

  Aunque este proyecto tiene un enfoque académico y no implica
  un despliegue productivo, esta fase se ha interpretado como la documentación sistemática y comunicación
  de los resultados obtenidos, incluyendo la elaboración de esta memoria técnica, la presentación de hallazgos
  clave sobre el potencial de los modelos Seq2Seq en el contexto clínico, y la formulación de recomendaciones
  específicas para futuros trabajos para facilitar su eventual adopción en entornos clínicos.

=== Comparación con otras metodologías

Para justificar la elección de CRISP-DM en este proyecto, es esencial analizar las características y
limitaciones de otras metodologías ampliamente utilizadas en el desarrollo de proyectos tecnológicos.
A continuación se presenta una comparación detallada con las metodologías Waterfall y Agile, examinando
su aplicabilidad específica en el contexto de proyectos de Machine Learning y ciencia de datos.
Metodología Waterfall (Cascada)

La metodología Waterfall representa el enfoque más tradicional y estructurado para la gestión de proyectos
de software. Desarrollada originalmente por Winston Royce en 1970, esta metodología sigue un proceso
secuencial y lineal donde cada fase debe completarse antes de proceder a la siguiente, sin posibilidad
de retroceso. Su característica fundamental es la necesidad de documentación exhaustiva en cada fase,
junto con una planificación detallada que define completamente los requisitos y el alcance al inicio
del proyecto.

En el contexto de este proyecto de predicción del avance del Parkinson, el modelo Waterfall presenta
limitaciones significativas que lo hacen inadecuado para proyectos de Machine Learning. La naturaleza
exploratoria de los datos requiere una exploración iterativa para comprender su estructura, calidad y
potencial predictivo, algo que el modelo Waterfall no permite ya que exige definir completamente los
requisitos antes de proceder. Además, la incertidumbre inherente en los proyectos de ML significa que
los resultados de una fase pueden invalidar completamente las asunciones de fases anteriores. Por ejemplo,
el análisis exploratorio puede revelar que los datos disponibles no son suficientes para el objetivo
planteado, requiriendo redefinir el problema inicial.

El desarrollo de modelos ML implica experimentación constante con diferentes algoritmos, hiperparámetros
y arquitecturas, una necesidad de iteración y refinamiento continuo que el modelo Waterfall no contempla.
A diferencia del desarrollo de software tradicional, donde los requisitos son relativamente estables,
en ML la calidad y disponibilidad de los datos puede cambiar dramáticamente la viabilidad del proyecto,
haciendo que el enfoque rígido de Waterfall sea contraproducente.

==== Metodología Agile

La metodología Agile surgió como respuesta a las limitaciones del modelo Waterfall, promoviendo un enfoque
iterativo e incremental para el desarrollo de software. Formalizada en el Manifiesto Agile de 2001, esta
metodología prioriza la adaptabilidad, la colaboración y la entrega temprana de valor. Su filosofía se
centra en el desarrollo iterativo mediante sprints cortos, la flexibilidad ante cambios en requisitos,
la colaboración continua entre stakeholders y la entrega temprana de valor al usuario.

Agile ofrece ciertas ventajas para proyectos de ML que lo hacen superior a Waterfall, particularmente
en su capacidad de adaptación basándose en los hallazgos obtenidos durante el análisis de datos. Su enfoque
en la iteración rápida facilita la experimentación con diferentes modelos y enfoques, mientras que el
feedback continuo permite validar hipótesis y ajustar el rumbo del proyecto frecuentemente.

Sin embargo, Agile también presenta limitaciones importantes cuando se aplica directamente a proyectos
de ML. La metodología no contempla explícitamente las fases críticas de comprensión y preparación de
datos, que pueden consumir gran parte del tiempo en un proyecto de ML. Por otra parte, el concepto de "entregable funcional"
se vuelve ambiguo en ML, donde un modelo puede funcionar técnicamente pero no tener valor predictivo
real. Las métricas tradicionales de Agile, como velocidad y burndown charts, no reflejan adecuadamente
el progreso en proyectos de investigación y experimentación.

Aunque Agile maneja bien los cambios en requisitos, no está diseñado para la incertidumbre fundamental
sobre la viabilidad técnica que caracteriza a los proyectos de ML.




En conclusión, para este proyecto específico de predicción del avance del Parkinson, CRISP-DM resulta la metodología
más apropiada por incluir fases dedicadas específicamente a la comprensión
y preparación de datos, aspectos críticos dada la complejidad de los datos clínicos y proteómicos; y por permitir
iteraciones entre fases pero de manera estructurada, evitando el riesgo de "deriva" en la experimentación
que puede ocurrir con enfoques menos estructurados.

== Planificación temporal

La planificación temporal de este proyecto se ha estructurado considerando que el TFG tiene una asignación
de 12 créditos ECTS, lo que equivale a 300 horas de trabajo según el Sistema Europeo de Transferencia
y Acumulación de Créditos (25 horas por crédito ECTS). Esta carga de trabajo se ha distribuido siguiendo
las seis fases de la metodología CRISP-DM, adaptándose a la disponibilidad variable durante el período
de trabajo, de la siguiente manera:

- _*Fase 1: Business Understanding*_ (Comprensión del negocio) - #emph-box[30 horas (10%)]

  - Análisis del contexto clínico del Parkinson: 10 horas
  - Revisión bibliográfica sobre ML aplicado a enfermedades neurodegenerativas: 12 horas
  - Estudio del desafío de Kaggle y análisis de soluciones previas: 8 horas

- _*Fase 2: Data Understanding*_ (Comprensión de los datos) - #emph-box[45 horas (15%)]

  - Exploración inicial del conjunto de datos: 20 horas
  - Análisis estadístico descriptivo: 15 horas
  - Evaluación de calidad de datos y valores faltantes: 10 horas

- _*Fase 3: Data Preparation*_ (Preparación de los datos) - #emph-box[60 horas (20%)]

  - Limpieza y transformación de datos: 15 horas
  - Implementación de técnicas de imputación: 20 horas
  - Construcción de secuencias temporales para modelos Seq2Seq: 25 horas

- _*Fase 4: Modeling*_ (Modelado) - #emph-box[90 horas (30%)]

  - Implementación de modelo baseline: 20 horas
  - Desarrollo de arquitecturas Seq2Seq (LSTM, GRU): 35 horas
  - Experimentación con hiperparámetros y regularización: 25 horas
  - Entrenamiento y ajuste de modelos: 10 horas

- _*Fase 5: Evaluation*_ (Evaluación) - #emph-box[30 horas (10%)]

  - Evaluación sistemática con métricas SMAPE: 15 horas
  - Análisis comparativo y validación de resultados: 15 horas

- _*Fase 6: Deployment*_ (Documentación y comunicación) - #emph-box[30 horas (10%)]

  - Redacción de la memoria técnica: 20 horas
  - Preparación de presentación y revisión final: 10 horas


Por otra parte se plantea una _*reserva para contigencias*_ de #emph-box[15 horas (5%)], reservadas para imprevistos
y revisiones adicionales. Tomando esta distribución como punto de partida, se tiene siempre en cuenta la
naturaleza emergente de los problemas en el dominio de la ciencia de datos reflejada por CRISP-DM, manteniendo la
posibilidad de redistribuir horas entre fases adyacentes en caso de necesidad.

#let time-table-headers = (
  ([Fase], [Horas], [%])
    .map(strong)
    .map(it => table.cell(it, align: center, fill: luma(230)))
)

#figure(
  align(
    center,
    table(
      columns: (40%, 20%, 10%),
      table.header(..time-table-headers),
      [Fase 1], [30h], [10%],
      [Fase 2], [45h], [15%],
      [Fase 3], [60h], [20%],
      [Fase 4], [90h], [30%],
      [Fase 5], [30h], [10%],
      [Fase 6], [30h], [10%],
      [Reserva para contingencias], [15h], [10%],
    ),
  ),
  caption: "Resumen de la distribución del tiempo total estimado para cada fase",
)

Para el control del progreso se lleva un registro de horas trabajadas en cada tarea identificada para mantener el control del presupuesto temporal.

=== Factores de riesgo:

Se han identificado ciertos factores de riesgo que pueden afectar al éxito de la planificación temporal en este proyecto.

- La preparación de datos puede requerir más tiempo del estimado debido a la complejidad de datos clínicos
- Los tiempos de entrenamiento de modelos pueden variar según la convergencia
- La documentación técnica puede extenderse si se requiere mayor profundidad en el análisis

== Presupuesto económico

El desarrollo de este proyecto de investigación en Machine Learning requiere una evaluación económica
que considere tanto los recursos humanos como los recursos de infraestructura necesarios para su ejecución.

=== Recursos humanos

El coste principal del proyecto corresponde al tiempo dedicado por el investigador principal.
La valoración se realiza considerando datos salariales actuales del sector tecnológico.

El rol que más se ajusta al perfil necesario para el desarrollo de este proyecto es el de un
científico de datos, particularmente calcularemos los costes salariales tomando en consideración un puesto de cientifico de datos sin
experiencia (Junior). Según los datos publicados en _Glassdoor_ #cite(<glassdoor2024>) recientemente los salarios brutos oscilan entre
los 22.000#sym.euro y 30.000#sym.euro. Tomando la estimación más alta aproximamos unos 15#sym.euro por hora trabajada:


#figure(
  align(center, [
    #table(
      columns: 2,
      align: left,
      table.cell([Horas totales], fill: luma(230)),
      [300h (12 créditos ECTS × 25 horas/crédito)],
      table.cell([Tarifa horaria], fill: luma(230)),
      [ 15€/h (Científico de datos Junior) ],
      table.cell([*Coste Total*], fill: luma(230)),
      [ *4.500€* ],
    )]),
  caption: "Prespuesto económico dedicado a recursos humanos.",
)

Se considera que todas las tareas a realizar pueden englobarse en este mismo rol, con lo que supondría el total de las horas estimadas
para la finalización del proyecto.

=== Recursos de infraestructura y de software

El presupuesto dedicado a infraestructura y software en este proyecto es pequeño en comparación con
el presupuesto dedicado a recursos humanos pero es importante detallarlo ya que en proyectos
del mismo ámbito el coste de infraestructura (normalmente en la nube) y las licencias de software anuales
son muy significativas.

En el contexto de este proyecto no se contemplan gastos debidos a licencias de software ya que
la totalidad de las herramientas elegidas para el desarrollo son de código abierto.

Por parte del hardware (infraestructura), el desarrollo se ha llevado a cabo en mi portatil personal.
El coste del portatil fue de 800#sym.euro. Tipicamente se considera amortizado un equipo informático en 4 años (25% anual).
Esta amortización considera un uso continuado del equipo, en mi caso al tratarse de mi equipo personal resulta más complicado
disociar el uso dedicado al proyecto del uso por otros asuntos propios, por ese mótivo en vez de dar la estimación del coste
amortizado usando años (o meses) se utilizarán las 300h planificadas. Para ello se tiene en cuenta el dato de que, a jornada
completa, se trabajan 1.826h anuales.


#figure(
  align(center, [
    #table(
      columns: 2,
      align: left,
      table.cell([Coste], fill: luma(230)),
      [800#sym.euro],
      table.cell([Horas de uso], fill: luma(230)),
      [ 300h ],
      table.cell([Horas amortización], fill: luma(230)),
      [ 7304h ],
      table.cell([Porcentaje imputable], fill: luma(230)),
      [ 4.1% ],
      table.cell([*Coste imputable*], fill: luma(230)),
      [ *32,8#sym.euro* ],
    )]),
  caption: "Prespuesto económico dedicado a recursos humanos.",
)

Otros gastos como los de internet y electricidad son especialmente dificiles de separar del consumo personal y no entrarán en este presupuesto.

=== Otros gastos

No se han identificado otras posibles fuentes de gastos ya que se optará por opciones disponibles abiertamente,
sin embargo se detallan aquí puntos que en otros proyectos similiares o en caso de extender el alcance de este,
podrían incurrir en gastos adicionales.

- Acceso a literatura científica: 0 € (acceso a través de la universidad)

- Acceso a los datos: 0#sym.euro (Disponibles abiertamente en el desafío de Kaggle)


#let money-table-headers = (
  ([Categoria], [Coste], [%])
    .map(strong)
    .map(it => table.cell(it, align: center, fill: luma(230)))
)

#figure(
  align(
    center,
    table(
      columns: (40%, 20%, 10%),
      table.header(..money-table-headers),
      [Recursos humanos], [4.500€], [99.3%],
      [Infraestructura], [32.8#sym.euro], [0.7%],
      [Software], [0#sym.euro], [0%],
      [Otros], [0#sym.euro], [0%],
      [*Total*], [4.532,8#sym.euro], [100%],
    ),
  ),
  caption: "Resumen del presupuesto",
)

== Ajuste a la realidad


Una vez finalizado el desarrollo del proyecto, es fundamental analizar el grado de ajuste entre la planificación inicial y la ejecución real, tanto en términos de tiempo como de recursos económicos.

=== Ajuste temporal

La planificación inicial contemplaba una dedicación total de *300 horas*. En la práctica, la dedicación real fue:

- Horas efectivamente dedicadas: #emph-box([310 horas])

Este ligero desfase (3,3%) se explica principalmente por tareas no previstas inicialmente, como la documentación
adicional e iteraciones validando y refinando el modelo. A pesar de superar el tiempo reservado para contingencias,
el proyecto se ha completado con un desfase mínimo, por lo que el ajuste temporal puede considerarse satisfactorio.

=== Ajuste económico

En cuanto al presupuesto económico, las desviaciones fueron mínimas y debidas también al tiempo adicional.

#figure(
  align(center, [
    #table(
      columns: (30%, 20%, 20%, 20%),
      table.header(
        table.cell([*Categoría*], align: center, fill: luma(230)),
        table.cell([*Presupuesto estimado*], align: center, fill: luma(230)),
        table.cell([*Coste real*], align: center, fill: luma(230)),
        table.cell([*Desviación*], align: center, fill: luma(230)),
      ),
      [Recursos humanos], [4.500#sym.euro], [4.650#sym.euro], [+150#sym.euro],
      [Infraestructura], [32,8#sym.euro], [33,9#sym.euro], [0#sym.euro],
      [Software], [0#sym.euro], [0#sym.euro], [0#sym.euro],
      [Otros], [0#sym.euro], [0#sym.euro], [0#sym.euro],
      [*Total*], [4.532,8#sym.euro], [4.683,9#sym.euro], [*+151.1#sym.euro*],
    )
  ]),
  caption: "Comparativa entre presupuesto estimado y real.",
)

=== Valoración general

El proyecto se ha desarrollado de forma eficiente, con desviaciones mínimas en tiempo y por lo tanto
en costes. El uso de herramientas *open source* ha sido clave para mantener el presupuesto bajo control,
y la amortización del equipo personal ha permitido limitar los costes de infraestructura.

Este ajuste a la realidad confirma la viabilidad económica del enfoque adoptado y sugiere que metodologías
similares pueden aplicarse en otros contextos académicos o incluso profesionales con recursos limitados.

