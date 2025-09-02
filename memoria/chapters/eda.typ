#import "../uva-template/template.typ": emph-box, error, ok

= Análisis exploratorio de datos

En esta sección se documenta el análisis exploratorio inicial realizado sobre los datos provistos por
el desafío de predicción de progresión del Parkinson. Primero, se explica la estructura y el contenido de los
datos según las descripciones aportadas por la fuente y se continúa por observaciones obtenidas en el cuaderno
de Jupyter incluido `eda.ipynb`.


== Terminología y contexto clínico

Un conocimiento, aunque superficial, de la terminología presente en la descripción del reto y el conjunto de datos
es importante ya que puede informar decisiones sobre el modelo a implementar. Es por esto que dedicamos una breve sección
a conceptos que se mencionan y que requieren una explicación adicional.
=== UPDRS

La *Unified Parkinson’s Disease Rating Scale* (UPDRS) es un instrumento clínico estandarizado que se
utiliza para cuantificar la progresión de los síntomas del Parkinson. Está compuesta por una serie de
ítems o preguntas que evalúan distintos aspectos de la enfermedad, agrupados en cuatro secciones. Cada
ítem se valora mediante una escala ordinal, y la suma de estos valores proporciona una medida de severidad
dentro de cada sección. Fue introducido por MDS (_Movement Disorder Society_) #cite(<updrs2007>), por lo que
en ocasiones también es referido por MDS-UPDRS.

Las puntuaciones posibles, así como los umbrales de severidad típicamente aceptados, se detallan a continuación:

- Parte I: Experiencias no motoras de la vida diaria (estado mental, humor, comportamiento).

- Parte II: Experiencias motoras de la vida diaria (actividades cotidianas como vestirse, alimentarse, etc.).

- Parte III: Examen motor realizado por personal médico (rigidez, temblores, reflejos posturales).

- Parte IV: Complicaciones motoras asociadas al tratamiento farmacológico (disquinesias, fluctuaciones).

Es importante destacar que cada parte tiene un rango de puntuaciones distinto ya que puede afectar al uso del dato
en este proyecto. Las puntuaciones más altas en cualquier parte indican una mayor severidad de los síntomas. Esta escala
permite capturar de forma sistemática la evolución de la enfermedad a lo largo del tiempo y evaluar la
eficacia de los tratamientos aplicados.



#figure(
  align(center, [
    #table(
      columns: (25%, 15%, 30%, 30%),
      align: center + horizon,
      table.header(
        table.cell([*Sección*], fill: luma(230), align: center),
        table.cell([*Rango total*], fill: luma(230), align: center),
        table.cell([*Leve*], fill: luma(230), align: center),
        table.cell([*Severa*], fill: luma(230), align: center),
      ),
      [Parte I], [0–52], [0–10], [≥22],
      [Parte II], [0–52], [0–12], [≥30],
      [Parte III], [0–132], [0–32], [≥59],
      [Parte IV], [0–24], [0–4], [≥13],
    )
  ]),
  caption: "Clasificación de la severidad según las puntuaciones UPDRS.",
)<updrs-ranges>

En el presente proyecto, las puntuaciones UPDRS constituyen las variables
objetivo que el modelo de aprendizaje automático debe predecir.

=== NPX

*NPX* (*Normalized Protein eXpression*) es una medida relativa de la abundancia de proteínas en una muestra
obtenida por técnicas de espectrometría de masas. Es una escala logarítmica donde una diferencia de
1 NPX representa una duplicación en la abundancia relativa (escala logarítmica en base 2). Este tipo de normalización permite comparar
muestras entre pacientes y visitas de manera robusta. No tiene unidades absolutas, lo que
limita la interpretación clínica directa, pero es útil en modelos comparativos y predictivos.

=== UniProt

UniProt es una base de datos biológica que proporciona información sobre secuencias y funciones de proteínas
En el conjunto de datos, cada proteína está identificada por un código único de UniProt (por ejemplo,
`P12345`), que permite consultar fácilmente su función biológica, localización celular y asociaciones
conocidas con enfermedades. Este identificador es clave para enriquecer el análisis con conocimiento
biológico previo o realizar anotaciones externas.

=== Proteínas y péptidos

Las *proteínas* son macromoléculas formadas por cadenas largas de aminoácidos. Están involucradas en
prácticamente todos los procesos biológicos. Las *péptidos* son fragmentos más cortos de proteínas; en
muchos casos, una proteína puede contener múltiples péptidos distintos.

== Descripción del conjunto de datos

El conjunto de datos proporcionado para el presente desafío tiene como objetivo predecir la progresión
de la enfermedad de Parkinson (EP) a partir de datos de abundancia proteica. Esta predicción se realiza
a partir de muestras de líquido cefalorraquídeo (LCR) analizadas mediante espectrometría de masas. Se
trata de un conjunto de datos longitudinal que incluye varias visitas por paciente a lo largo de varios
años, acompañadas de evaluaciones clínicas del estado de la enfermedad.

Los datos están organizados en cuatro archivos principales:

=== `train_peptides.csv`

Contiene mediciones de espectrometría de masas a nivel de péptido, es decir, subcomponentes de proteína
. Cada fila representa la abundancia de un péptido específico en una muestra determinada.

#figure(
  align(center, [
    #table(
      columns: (25%, 75%),
      table.header(
        table.cell([*Campo*], fill: luma(230)),
        table.cell([*Descripción*], fill: luma(230)),
      ),
      [_visit_id_], [Identificador único de la visita.],
      [_visit_month_], [Mes relativo desde la primera visita del paciente.],
      [_patient_id_], [Identificador del paciente.],
      [_UniProt_], [Código UniProt de la proteína asociada.],
      [_Peptide_], [Secuencia de aminoácidos del péptido.],
      [_PeptideAbundance_], [Frecuencia relativa del péptido en la muestra.],
    )]),
  caption: "Descripción de los campos de los datos de péptidos.",
)

=== `train_proteins.csv`

Agrega la información del archivo anterior a nivel proteico. Es decir, contiene la abundancia de proteínas obtenida a partir de los péptidos componentes.

#figure(
  align(center, [
    #table(
      columns: (15%, 85%),
      table.header(
        table.cell([*Campo*], fill: luma(230)),
        table.cell([*Descripción*], fill: luma(230)),
      ),
      [_visit_id_], [Identificador único de la visita.],
      [_visit_month_], [Mes relativo desde la primera visita del paciente.],
      [_patient_id_], [Identificador del paciente.],
      [_UniProt_], [Código UniProt de la proteína.],
      [_NPX_], [Abundancia normalizada de la proteína.],
    )]),
  caption: "Descripción de los campos de los datos de proteínas.",
)

=== `train_clinical_data.csv`

Contiene las evaluaciones clínicas asociadas a cada visita en la que se recogió una muestra de LCR. Los
campos incluyen información sobre la puntuación del paciente en diferentes partes de la escala UPDRS
(*Unified Parkinson's Disease Rating Scale*).

#figure(
  align(
    center,
    [
      #table(
        columns: (30%, 70%),
        table.header(
          table.cell([*Campo*], fill: luma(230)),
          table.cell([*Descripción*], fill: luma(230)),
        ),
        [`visit_id`], [Identificador único de la visita.],
        [`visit_month`], [Mes relativo desde la primera visita del paciente.],
        [`patient_id`], [Identificador del paciente.],
        [`updrs_1` - `updrs_4`],
        [Puntuaciones en las partes 1 a 4 de la escala UPDRS.],

        [`upd23b_clinical_`
          `state_on_medication`],
        [Indica si el paciente estaba bajo medicación (como Levodopa) durante la evaluación.],
      )],
  ),
  caption: "Descripción de los campos de los datos de pacientes.",
)

=== `supplemental_clinical_data.csv`

Este archivo contiene registros clínicos adicionales de pacientes que no tienen muestras asociadas de LCR. Se utiliza para proporcionar contexto adicional sobre la progresión típica de la enfermedad. Su estructura es exactamente igual que la del fichero `train_clinical_data.csv`.

=== Consideraciones generales

A partir de la descripción provista de los datos destacamos las siguientes consideraciones que han influido en el análisis.

- La naturaleza longitudinal del conjunto de datos y la estructura jerárquica (paciente,visita,medición) exigen un tratamiento cuidadoso del tiempo como variable.
- Existe una alta dimensionalidad en las mediciones proteicas, tanto a nivel de péptido como de proteína. Es decir para una única visita tenemos muchas proteínas y
  peptidos distintos que nos proporcionan un dato para esa visita.
- Es necesario tratar cuidadosamente en particular los datos de UPDRS y NPX ya que tienen una escalas particulares, asegurando que los valores están normalizados de manera
  interpretable por el modelo.

== Observaciones realizadas

En este apartado se comentan las observaciones obtenidas en el cuaderno incluido `eda.ipynb`, comentando además el código utilizado.

=== Carga

#align(
  center,
  figure(
    ```python
    import os
    import pandas as pd
    import numpy as np
    import seaborn as sns
    data_path = "amp-parkinsons-disease-progression-prediction"
    supplemental = pd.read_csv(os.path.join(data_path, "supplemental_clinical_data.csv"))
    patient =pd.read_csv(os.path.join(data_path, "train_clinical_data.csv"))
    peptides = pd.read_csv(os.path.join(data_path, "train_peptides.csv"))
    proteins = pd.read_csv(os.path.join(data_path, "train_proteins.csv"))
    ```,
    caption: "Carga de los datos usando pandas.",
  ),
)

La carga se ha realizado en _DataFrames_ de `pandas`, lo tomaremos como punto de partida tanto para este
análisis como para el modelo final.

=== Cardinalidad de los datos


#align(
  center,
  figure(
    ```python
    print(f"""Cardinalidad de los datos:
    Hay {len(proteins.UniProt.unique())} proteínas únicas
    Hay {len(peptides.Peptide.unique())} péptidos únicos
    Hay {len(peptides[["Peptide", "UniProt"]].drop_duplicates())} pares de proteina-peptido únicos.
    Hay {len(patient.patient_id.unique())} pacientes
    Hay {len(patient)} visitas
    Hay {len(supplemental.patient_id.unique())} pacientes (suplementario)
    Hay {len(supplemental)} visitas (suplementario)
    """)
    ```,
    caption: "Obtención de datos de cardinalidad",
  ),
)

Se realiza un recuento de los datos disponibles a distintos niveles, el dato central para el modelo es el de "visita".

#figure(
  align(center, [
    #table(
      columns: 2,
      align: left,
      table.cell([*Métrica*], fill: luma(230)),
      table.cell([*Valor*], fill: luma(230)),

      [Proteínas únicas], [227],
      [Péptidos únicos], [968],
      [Péptido-proteína únicos], [968],
      [Pacientes (principal)], [248],
      [Visitas (principal)], [2615],
      [Pacientes (sulementario)], [771],
      [Visitas (suplementario)], [2223],
    )
  ]),
  caption: "Cardinalidad del conjunto de datos.",
)

Como podemos observar el modelo podrá disponer para su entrenamiento del orden de 2600 datos de entrada
completos, con mediciones de proteínas o hasta 5800 con mediciones de UPDRS. Son números no muy grandes
y esto representa uno de los principales riesgos del proyecto ya que la cantidad de los datos de entrado
disponibles inlfuyen dramáticamente en la calidad y utilidad predictiva del modelo final. Por otra parte
se puede observar que no hay peptidos en común para distintas proteínas.

=== Validación de las relaciones

#figure(
  align(
    center,
    [
      ```python
      # Comprobamos que efectivamente es una clave primaria
      print("¿Hay algún visit_id duplicado en clinical?")
      print(patient["visit_id"].duplicated().any())
      print("¿Hay algún visit_id duplicado en supplemental?")
      print(supplemental["visit_id"].duplicated().any())

      # Comprobaciones de relación entre tablas

      # Comprobación de la relación 1 a 1
      print("¿Están todos los visit_id de proteínas en 'clinical'?")
      difference = set(proteins["visit_id"]).difference(set(patient["visit_id"]))
      print(not difference)
      if difference:
          print(len(difference))

      print("¿Están todos los visit_id de 'clinical' en proteínas?")
      difference = set(patient["visit_id"]).difference(set(proteins["visit_id"]))
      print(not difference)
      if difference:
          print(len(difference))

      print(
          "¿Es cierto que los datos de 'supplemental' no tienen datos de proteínas asociados?"
      )
      intersection = set(supplemental["visit_id"]).intersection(proteins["visit_id"])
      print(not intersection)
      if intersection:
          print(len(intersection))

      # Comprobación de la relación 1 a n
      print("¿Estan todas las mediciones de peptidos asociadas a una medicion de proteínas?")
      difference = set(peptides[["visit_id", "UniProt"]].apply(tuple, axis=1)).difference(
          proteins[["visit_id", "UniProt"]].apply(tuple, axis=1)
      )
      print(not difference)
      if difference:
          print(len(difference))
      ```
    ],
  ),
  caption: "Validaciones de las relaciones.",
)


Es importante validar las suposiciones que hacemos sobre los datos, evitando sostener ninguna suposición implícita
que pueda producir errores más adelante difíciles de depurar. Unas de estas suposiciones, que se forman de manera inmediata
a partir de la descripción de los datos, es la de las relaciones entre las tablas y qué campos forman claves primarias y claves foráneas.

#let correct-box = emph-box(color: ok)[Correcto]
#let incorrect-box = emph-box(color: error)[Incorrecto]

- ¿Hay algún `visit_id` duplicado en clinical? #correct-box

  No lo hay, `visit_id`, formado por `patient_id` y `visit_month` es una clave primaria.

- ¿Hay algún `visit_id` duplicado en supplemental? #correct-box

  No lo hay. Lo mismo aplica para el conjunto de datos suplementario.

- ¿Están todos los visit_id de proteínas en 'clinical'? #incorrect-box

  Existen 45 visitas distintas cuyas mediciones en la tabla de proteínas no se corresponden con ninguna
  visita que figure en la tabla principal. Estos datos tendrán que ser descartados para el modelo final.

- ¿Están todos los visit_id de 'clinical' en proteínas? #incorrect-box

  Existen 1547, es decir cerca de la mitad de los datos no dispone de mediciones de
  proteínas. Se deberá decir como tratar tantos registros con valores faltantes, ya que eliminarlos por completo reduciría mucho el conjunto de datos disponible.

- ¿Los datos de 'supplemental' no tienen datos de proteínas asociados? #correct-box

- ¿Estan todas las mediciones de peptidos asociadas a una medicion de proteínas? #correct-box

=== Análisis de la distribución de los datos de proteínas

#figure(
  align(
    center,
    [
      ```python
      sns.set_theme(rc={'figure.figsize':(11.7,8.27)})
      proteins["logNPX"] = np.log2(proteins["NPX"])
      unique_proteins = proteins["UniProt"].unique()
      protein = unique_proteins[1]
      some_proteins = proteins.loc[proteins["UniProt"] == protein]
      sns_plot = sns.violinplot(some_proteins, x="logNPX", y="UniProt")
      sns_plot.get_figure().savefig("log_npx_violin_plot.png")
      plt.close()
      sns_plot = sns.violinplot(some_proteins, x="NPX", y="UniProt")
      sns_plot.get_figure().savefig("npx_violin_plot.png")
      ```
    ],
  ),
  caption: [Creación de gráficos de violin para los valores NPX usando _seaborn_.],
)

Para examinar las distribuciones de las distintas proteínas se utilizan _violin plots_,
un _violin plot_ es una representación gráfica que combina un diagrama de caja (boxplot) con una estimación
de densidad de núcleo (KDE), con el objetivo de visualizar simultáneamente estadísticas resumidas y la
distribución completa de una variable continua. En su forma, el gráfico se asemeja a un violín: la anchura
de cada sección representa la densidad de probabilidad estimada en ese rango de valores, permitiendo
identificar características como asimetrías, modas múltiples o colas largas en la distribución. Superpuesto
a esta densidad, el gráfico incluye elementos tradicionales del boxplot como la mediana, marcada dentro de la caja;
los cuartiles, que determinan el tamaño de la caja; y los bigotes, la linea que se extiende desde la caja marcando los límites a partir
de los cuales un valor pasa a considerarse extremo.


#align(center)[
  #figure(
    image("../assets/npx_violin_plot.png", width: 50%),
    caption: [Diagrama de violín de la distribución de la medida NPX para una proteina concreta],
  )
]

En primer lugar podemos observar la escala (en el orden de $10^4$ en este caso pero con gran variación entre proteínas), esto nos indica que a pesar de que NPX normalmente se
utiliza en escala logarítmica (normalmente entre 10 y 20), en este caso tenemos los valores "linearizados".


#align(center)[
  #figure(
    image("../assets/log_npx_violin_plot.png", width: 50%),
    caption: [Diagrama de violín utilizando $ log_2 text("NPX") $],
  )
]

Observando la distribución de esta y otras proteínas vemos que se tratan en general de distribuciones unimodales y simétricas:
similares a la distribución normal.

=== Análisis de la distribución de los datos UPDRS

#figure(
  align(
    center,
    [
      ```python
      updrs_cols = [f"updrs_{i}" for i in range(1,5)]
      updrs = pd.concat([patient[updrs_cols], supplemental[updrs_cols]])
      sns_plot = sns.violinplot(updrs)
      sns_plot.get_figure().savefig("updrs_violins.png")
      updrs_ranges = [52, 52, 132, 24]

      for i in range(1,5):
          updrs[f"updrs_{i}"] = updrs[f"updrs_{i}"] / updrs_ranges[i-1]
      sns_plot = sns.violinplot(updrs)
      sns_plot.get_figure().savefig("norm_updrs_violins.png")
      ```
    ],
  ),
  caption: [Creación de gráficos de violin para los datos de UPDRS usando _seaborn_.],
)

Se observa en general distribuciones con peso cerca del 0, siendo los valores mayores progresivamente
más raros. Esto es especialmente notable en la categoríá IV, donde la mayoría de los valores son 0 o
cercanos al 0. Cabe señalar también la bimodalidad de la distribución del los datos de la categoría II
.

#figure(
  grid(
    columns: 2,
    // dos columnas auto-ajustadas
    gutter: 2mm,
    // espacio entre imágenes
    image("../assets/updrs_violins.png"),
    image("../assets/norm_updrs_violins.png"),
  ),
  caption: [Distribución de los valores UPDRS, incluyendo datos del dataset "clínico" y "suplementario". A la izquierda
    distribución de los datos sin normalizar, a la derecha distribución de los datos normalizados según
    los rangos especificados por categoría (#ref(<updrs-ranges>)).],
)


#figure(
  align(
    center,
    [
      ```python
      for col in updrs_cols:
          print(f"{col}: {len(updrs[updrs[col].isna()])} / {len(updrs)} ({len(updrs[updrs[col].isna()]) * 100/ len(updrs):.2f}%)")
      ```
    ],
  ),
  caption: [Obtención de información sobre los valores nulos.],
)

Se a realizado también un estudio de los valores faltantes en los datos de UPDRS, destacando un altísimo
porcentaje de valores faltantes en la categoría IV. En vista de este análisis se juzga sensato imputar
a estos casos el valor 0 para su uso en el modelo.

#figure(
  align(center, [
    #table(
      columns: 3,
      align: left,
      table.cell([*Categoría*], fill: luma(230)),
      table.cell([*Número de valores nulos*], fill: luma(230)),
      table.cell([*Número de valores nulos*], fill: luma(230)),
      [*I*], [214], [4.42%],
      [*II*], [216], [4.46%],
      [*III*], [30], [0.62%],
      [*IV*], [1966], [40.64%],
    )
  ]),
  caption: "Número de valores faltantes para los datos de UPDRS",
)

=== Distribución de mes de visita

#figure(
  align(
    center,
    [
      ```python
      visit_months = pd.concat([patient["visit_month"], supplemental["visit_month"]])
      sns_plot = sns.histplot(visit_months, binwidth=1)
      sns_plot.set_xticks(visit_months.unique())
      sns_plot.get_figure().savefig("visit_month_hist.png")
      ```
    ],
  ),
  caption: [Creación de histograma de los meses de visita.],
)

En el siguiente histograma se aprecia la distribución de los meses de visita. Podemos realizar varias observaciones: en primer lugar
la primera visita (mes 0) es la más frecuente, de la que más datos disponemos. Esto puede haber dado lugar a que los valores bajos de UPDRS
sean los más frecuentes, entendiendo que empeoran con el tiempo. Por otra parte, la frecuencia de visita es tipicamente de 6 meses. Sin embargo,
tenemos valores para meses 3, 5 y 9, y las visitas más tardías parecen tener frecuencia anual. Esto tendrá consecuencias en el tratamiento de datos
para su uso en un modelo ya que habrá que decidir entre aceptar una frecuencia variable o ajustar o descartar los valores que no encajen en la frecuencia
semestral.

#align(center)[
  #figure(
    image("../assets/visit_month_hist.png", width: 80%),
    caption: [Histograma de los meses de visita.],
  )
]


=== Evolución temporal

#figure(
  align(
    center,
    [
      ```python
      for col in updrs_cols:
          sns_plot = sns.lineplot(
              pd.concat([patient, supplemental], axis=0, ignore_index=True).fillna(0),
              x="visit_month",
              y=col,
              estimator="mean",
              errorbar=("ci", 95),
          )
          sns_plot.get_figure().savefig(f"evolution_{col}.png")
          plt.close()
      ```
    ],
  ),
  caption: [Creación de gráficos de evolución temporal],
)

Para estudiar la evolución en las distintas categorías de UPDRS a lo largo del tiempo se ha graficado la media
por mes de visita incluyendo un área de color alrededor que representa el intervalo de confianza para la media
con confianza $0.95$. De esta manera el área se ensancha si hay mayor variación o menos datos.

Se puede observar en todo caso una tendencia a aumentar en el tiempo, aunque el incremento no es dramático.
En la categoría IV se observa un incremento pero en todo caso esta categoría se mantiene con valores muy bajos
(recordamos que es la correspondiente a los efectos adversos de la medicación).

#figure(
  grid(
    columns: 2,
    // dos columnas auto-ajustadas
    gutter: 2mm,
    // espacio entre imágenes
    image("../assets/evolution_updrs_1.png"),
    image("../assets/evolution_updrs_2.png"),

    image("../assets/evolution_updrs_3.png"),
    image("../assets/evolution_updrs_4.png"),
  ),
  caption: [Evolución media a lo largo del tiempo.],
)

En la próxima sección validaremos esta observación encontrando una correlación positiva entre los valores UPDRS y el tiempo.

=== Correlación cruzada

#figure(
  align(
    center,
    [
      ```python
      df = pd.concat([patient, supplemental], axis=0, ignore_index=True).fillna(0)
      df = df.rename(columns={"upd23b_clinical_state_on_medication": "on_medication"})

      df["on_medication"] = (
          df["on_medication"]
          .case_when(
              [
                  (df.on_medication.eq("On"), 1),
                  (df.on_medication.eq("Off"), -1),
              ]
          )
          .fillna("0")
      )
      df = df.drop(columns=["patient_id", "visit_id"])
      cross_corr_matrix = df.corr()

      plt.figure(figsize=(20, 16))
      sns.heatmap(
          cross_corr_matrix,
          annot=False,
          cmap="coolwarm",
          cbar=True,
          square=True,
          xticklabels=True,
          yticklabels=True,
          linewidths=0.1,
      )

      plt.title("Matriz de correlación cruzada", fontsize=16)
      plt.xticks(fontsize=20, rotation=90)
      plt.yticks(fontsize=20)
      plt.tight_layout()
      plt.savefig("correlation.png")
      ```
    ],
  ),
  caption: [Creación de una representación de mapa de calor de la matriz de correlación.],
)

Se utiliza una representación de mapa de calor de la matriz de correlación, obtenida calculando el coeficiente de correlación de Pearson dado por

#align(center)[
  $rho_(X, Y) = frac(text("cov")(X,Y), sigma_X sigma_Y)$]

El coeficiente de correlación de Pearson es la covarianza normalizada a un intervalo de $[-1,1]$. Captura *exclusivamente* la relación lineal entre 2
variables, esto significa que el coeficiente de Pearson puede ser bajo para dos variables que sí están relacionadas si su relación es no lineal.


#align(center)[
  #figure(
    image("../assets/correlation.png", width: 85%),
    caption: [Mapa de calor de la matriz de correlación.],
  )
]

Se observa que las correlaciones ms importantes son entre los valores en las distintas categorías de
UPDRS, que tienden a crecer a la vez. No obstante, la relación con el mes de visita no es aparente utilizando
esta métrica. Con el objetivo de ilustrar más estas correlaciones encontradas, observamos también los siguientes _scatter plots_,
donde, por pares, se tiene en cada eje una de las variables UPDRS. La relación lineal es evidente de esta manera.



#align(center)[
  #figure(
    image("../assets/pairplot.png", width: 85%),
    caption: [_Pair plot_ para las variable UPDRS. La diagonal es un histograma de cada una de las variables, mientras que en el
      resto de celdas se tiene un _scatter plot_ con las variables correspondientes en los ejes.],
  )
]

