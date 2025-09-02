= Modelo basado en una red neuronal simple

En este capítulo desarrollaremos un modelo sencillo basado en una red neuronal simple, que tomaremos como base con la que
comparar nuestra solución. Gran parte del preprocesado de los datos para preparar las características que tomará el modelo
como entrada (_feature engineering_) será común al utilizado finalmente para el modelo Seq2Seq, con lo que se entrará en detalle
en esa parte.

Este modelo esta inspirado en las mejores soluciones para el reto original, varias de entre ellas basadas en redes neuronales. Estas
soluciones tienen también en común que ignoran por completo los datos de proteínas y péptidos, ya que consistentemente encontraron que
introducía más ruido y que la mayor cantidad de información se encontraba en columnas como el mes de visita o si para una visita se tiene
información de proteínas. En este capítulo intentamos también replicar estos hallazgos.

Recordamos que el objetivo es dar una predicción a 6, 12 y 24 meses en cada una de las cuatro categorías UPDRS.

== Preprocesado de los datos

En esta sección se describe como se ha llevado a cabo el tratamiento de los datos para finalmente entrenar el modelo base. Este es un
paso esencial en el proceso del que depende gran parte del éxito o fracaso de un modelo. Se ha dividido en dos partes:

- *Limpieza de datos*: Tratamiento de valores nulos, normalización, ...
- *Creación de características objetivo*: En este caso las características objetivo son los valores UPDRS pasado cierto número de meses, estos datos
  no están presentes directamente en los datos de entrada y se deben obtener en esta etapa de preprocesado.
- *Creación de características de entrada auxiliares*: Este paso está especialmente inspirado en las soluciones más exitosas del reto que señalan el
  uso de características como mes de la última visita o la presencia de datos proteómicos.

Partimos de haber cargado los datos de igual manera que en el capítulo anterior en _DataFrames_ de _pandas_.

=== Limpieza de datos

El primer paso es normalizar los valores a utilizar. En el caso de los valores de péptidos y proteínas
primero utilizamos los valores en escala logaritmica para la métrica NPX y posteriormente los normalizamos
utilizando el rango (_min max scaling_).

En el caso de los datos referentes a UPDRS, trataremos de igual modo a los procedentes de ambos conjuntos de datos teniendo en cuenta en cualquier caso
que pueden no tener una medición de proteínas y péptidos asociada. La normalización de los valores UPDRS se realiza de nuevo
utilizando los valores máximos especificados en la definición de la escala.

#align(
  center,
  figure(
    ```python
    scaled_patient = pd.concat([patient,supplemental])
    updrs_ranges = [52,52,132,24]
    updrs_cols = [f"updrs_{i}" for i in range(1,5)]
    for updrs_range, col in zip(updrs_ranges, updrs_cols):
        scaled_patient[col] /= updrs_range

    scaled_protein = proteins.copy()
    scaled_protein["NPX"] = np.log2(proteins["NPX"])
    scaled_protein = (
        scaled_protein[["UniProt", "NPX"]]
        .groupby("UniProt")
        .agg(["min", "max"])
        .droplevel(0, axis=1)
        .join(proteins.set_index("UniProt"))
    )
    scaled_protein["NPX"] = (
      (scaled_protein["NPX"] - scaled_protein["min"]) /
      (scaled_protein["max"] - scaled_protein["min"])
    ).drop(columns=["min", "max"])

    scaled_peptide = peptides.copy()
    scaled_peptide["PeptideAbundance"]= np.log2(peptides["PeptideAbundance"])
    scaled_peptide = (
        scaled_peptide[["UniProt", "PeptideAbundance", "Peptide"]]
        .groupby(["UniProt", "Peptide"])
        .agg(["min", "max"])
        .droplevel(0, axis=1)
        .join(peptides.set_index(["UniProt", "Peptide"]))
    )
    scaled_peptide["PeptideAbundance"] = (
      (scaled_peptide["PeptideAbundance"] - scaled_peptide["min"]) /
      (scaled_peptide["max"] - scaled_peptide["min"])
    ).drop(columns=["min", "max"])
    ```,
    caption: "Normalización de datos de proteínas, péptidos y UPDRS.",
  ),
)

Además como parte de la limpieza de los datos se transforma la columna que indica si un paciente esta o no usando medicación en una columna númerica, ya que es provista como cadena
y por lo tanto no sería utilizable por el modelo. Dado que realmente se tienen 3 valores posibles en
esta columna, "On", "Off" y valor nulo, se ha decidido hacer corresponderles respectivamente 1, -1 y
0. De esta manera a la falta de información sobre el estado del paciente con respecto a la medicación
se le asigna un valor númerico intermedio. El objetivo es poder representar de manera interpretable esta
falta de información al modelo ya que en ningún caso puede tratar con valores nulos directamente.

#align(
  center,
  figure(
    ```python
    scaled_patient = scaled_patient.rename(
        columns={"upd23b_clinical_state_on_medication": "on_medication"}
    )
    scaled_patient["on_medication"] = (
        scaled_patient["on_medication"]
        .case_when(
            [
                (scaled_patient.on_medication.eq("On"), 1),
                (scaled_patient.on_medication.eq("Off"), -1),
            ]
        )
        .fillna("0")
    )
    ```,
    caption: "Tratamiento de la variable que representa si el paciente está, en el momento de la visita, tomando alguna medicación.",
  ),
)

=== Creación de características objetivo

Se identifican 12 característias objetivo, correspondientes del producto entre los 3 instantes de tiempo requeridos (dentro de 6, 12 y 24 meses) y las
4 categorías de la escala UPDRS consideradas.


#align(
  center,
  figure(
    ```python
    from itertools import product

    def safe_get(patient_id, visit_month, target_col):
        try:
            return indexed_scaled_patient.loc[(patient_id, visit_month), [target_col]].iloc[
                0
            ]
        except KeyError:
            return np.nan

    with_leads = scaled_patient
    indexed_scaled_patient = scaled_patient.set_index(["patient_id", "visit_month"])

    for plus_months, target_col in product(
        [6, 12, 24],
        [
            "updrs_1",
            "updrs_2",
            "updrs_3",
            "updrs_4"
        ],
    ):
        with_leads[f"{target_col}_plus_{plus_months}"] = with_leads.apply(
            lambda row: safe_get(
                row["patient_id"],
                row["visit_month"] + plus_months,
                target_col
            ),
            axis=1,
        )
    with_leads = with_leads[~with_leads.updrs_1_plus_6.isna()]
    ```,
    caption: "Creación de características objetivo",
  ),
)

En este fragmento de código se limpian también aquellas visitas que NO tienen datos dentro de 6 meses. Esta decisión elimina una cantidad notable de datos,
ya que como vimos en el análisis exploratorio hay visitas que no se ajustan a la cadencia semestral. Por otra parte simplifica significativamente el modelo.

=== Creación de características de entrada auxiliares

En las soluciones más exitosas dentro del contexto del desafío original se encontró mucha utilidad a
características asociadas a las decisiones médicas tomadas en cada caso, que como ya hemos señalado mina la utilidad
práctica real del modelo que debería informar estas mismas decisiones. Sin embargo, al formar parte de muchas de
las mejores soluciones, se ha decidido incluirlas en este modelo base.

Las características incluidas son:

- Si existen datos proteómicos para una visita, indicando que el médico responsable consideró adecuado que se realizara un análisis.
- Mes de ultima visita.
- Total de visitas.
- Meses desde la última visita.

#align(
  center,
  figure(
    ```python
    with_leads = with_leads.set_index(["patient_id", "visit_month"]).join(
        proteins[["patient_id", "visit_month", "NPX"]]
        .groupby(["patient_id", "visit_month"])
        .count(),
        how="left",
    ).reset_index()
    with_leads["did_test"] = with_leads["NPX"].case_when([(with_leads["NPX"] > 0, 1)]).fillna(0)
    with_leads = with_leads.drop(columns=["NPX"])
    ```,
    caption: "Creación de una característica de entrada auxiliar representando si existen datos proteómicos para la visita",
  ),
)


#align(
  center,
  figure(
    ```python
    with_leads["last_visit"] = with_leads.sort_values(by=['patient_id', 'visit_month']).groupby("patient_id")["visit_month"].shift(1).fillna(0)
    with_leads["visit_diff"] = with_leads["visit_month"] - with_leads["last_visit"]
    with_leads["visit_count"] = with_leads.groupby('patient_id').cumcount()
    ```,
    caption: "Creación de características de entrada auxiliares representando el número de visitas hasta la visita dada y la diferencia en meses entre la visita actual y la anterior.",
  ),
)

== Entrenamiento del modelo

El primer paso para pasar a entrenar el modelo será crear el conjunto de datos de entrada y el conjunto de datos de salida seleccionando
las características adecuadas que hemos estado creando. Además pasamos de un `DataFrame` en `pandas` a arrays de `numpy`.

#align(
  center,
  figure(
    ```python
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from sklearn.model_selection import KFold

    no_na = with_leads.fillna(0)
    feature_cols = [
        "visit_month",
        "did_test",
        "on_medication",
        "updrs_1",
        "updrs_2",
        "updrs_3",
        "visit_count",
        "visit_diff",
        "updrs_4"
    ]
    target_cols = [
        f"{target_col}_plus_{plus_months}"
        for plus_months, target_col in product(
            [6, 12, 24],
            [
                "updrs_1",
                "updrs_2",
                "updrs_3",
                "updrs_4"
            ],
        )
    ]
    X = no_na[feature_cols].to_numpy(dtype="float")
    y = no_na[target_cols].to_numpy(dtype="float")

    ```,
    caption: "Creación de los conjuntos de entrada y de salida",
  ),
)

Posteriormente definiremos la métrica SMAPE que intentaremos minimizar durante el entrenamiento:


$
  text("SMAPE") = (100 / n) sum_(t=1)^n abs(hat(y)_t - y_t) / ( (abs(y_t) + abs(hat(y)_t)) \/ 2 )
$

Se pueden definir métricas personalizadas mediante funciones. Destacamos de la definición dos puntos:

- No se calcula la métrica como un porcentaje. Como se va a optimizar el módelo buscando un mínimo para
  esta función, el resultado es el mismo si no multiplicamos por 100.

- Se añade un sumando muy pequeño al denominador para evitar la división por 0 en el caso en que tanto el valor real
  como el valor que se ha obtenido son 0.

#align(
  center,
  figure(
    ```python
    def smape(y_true, y_pred):
        epsilon = 1e-10
        numerator = tf.abs(y_true - y_pred)
        denominator = tf.abs(y_true) + tf.abs(y_pred) + epsilon
        smape = 2 * numerator / denominator
        return tf.reduce_mean(smape)
    ```,
    caption: "Definición de la función de pérdida: SMAPE",
  ),
)

Definimos ahora el modelo. Se han probado distintas definiciones de tamaños y número de capas intermedias, coeficiente de _dropout_ y
funciones de activación con resultados similares.

#align(center, figure(
  ```python
  def create_model():
      model = Sequential(
          [
            Dense(64, activation="relu"),
            Dropout(0.1),
            Dense(32, activation="relu"),
            Dropout(0.1),
            Dense(16, activation="relu"),
            Dropout(0.1),
            Dense(12, activation="linear"),
          ]
      )
      model.compile(optimizer="adam", loss=smape, metrics=["mae"])
      return model
  ```,
  caption: "Definición del modelo",
))Este modelo está diseñado para un problema de regresión con 9 características de entrada y 12 de salida.
La capa de entrada se omite en la definición.La red tiene tres capas ocultas (las intermedias entre entrada y salida)
con 64, 32 y 16 neuronas respectivamente. Esta es una forma típica de red neuronal, en embudo (_funnel_).
cada una con activación ReLU para capturar no linealidades.

Se aplica un _dropout_ del 10% después de cada capa oculta para combatir el *sobreajuste* (_overfitting_). El dropout funciona
“apagando” aleatoriamente un porcentaje de neuronas durante el entrenamiento, forzando a la red a no
depender demasiado de ninguna unidad en particular. Esto promueve una representación más robusta y dispersa
del conocimiento, mejorando la capacidad del modelo para generalizar a datos nuevos, especialmente cuando
la cantidad de datos es limitada.

Por último, el modelo se compila con el optimizador Adam  Adam es una variante avanzada del descenso de gradiente que adapta la tasa de aprendizaje
(cuanto pueden variar los coeficientes en cada paso de entrenamiento) para cada parámetro de forma automática.
Esto acelera la convergencia y reduce la necesidad de ajustes manuales.


#align(center)[
  #figure(
    image("../assets/ReLu.png", width: 50%),
    caption: [La función $"ReLu"(x) = max(x, 0)$],
  )
]

Pasamos al entrenamiento como tal del modelo, una vez hemos definido el modelo usando `pytorch` resulta tan sencillo
como llamar al método fit con los datos y parámetros adecuados.

Se utiliza _K-Fold cross-validation_ para evaluar de forma más robusta el desempeño del modelo, especialmente
cuando se cuenta con un conjunto de datos pequeño. En lugar de entrenar y validar una sola vez con una
partición fija, K-Fold divide los datos en varias “folds” y entrena el modelo en diferentes combinaciones
de entrenamiento y validación. Esto ayuda a reducir la varianza en la estimación del error y asegura
que el modelo generalice bien a distintos subconjuntos del conjunto de datos, minimizando el riesgo de
sobreajuste a una partición particular. Además, proporciona una métrica más confiable y estable del rendimiento
real, lo cual es fundamental para tomar decisiones informadas sobre ajustes o mejoras del modelo.

Las épocas (_epochs_) representan cuántas veces el modelo va a recorrer todo el conjunto de entrenamiento durante
el aprendizaje. Más epochs permiten al modelo ajustar mejor sus pesos, pero demasiados pueden causar
sobreajuste, especialmente con datos pequeños. Por otro lado, el tamaño de lote (_batch size_) indica cuántas muestras procesa
el modelo antes de actualizar sus parámetros en cada paso del entrenamiento. Un tamaño de lote pequeño, como
16, hace que las actualizaciones sean más frecuentes y ruidosas, lo que puede ayudar a escapar de mínimos
locales y mejorar la generalización, aunque aumenta el tiempo de entrenamiento.
#align(
  center,
  figure(
    ```python
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = []
    for train_idx, val_idx in kf.split(X):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        model = create_model()
        history = model.fit(
            X_train_fold,
            y_train_fold,
            validation_data=(X_val_fold, y_val_fold),
            epochs=50,
            batch_size=16,
            verbose=1,
        )

        val_loss = model.evaluate(X_val_fold, y_val_fold, verbose=0)[0]
        cv_results.append(val_loss)

    print(
        f"Cross-Validation SMAPE Loss: {np.mean(cv_results):.4f} ± {np.std(cv_results):.4f}"
    )
    ```,
    caption: "Entrenamiento del modelo, incluyendo los resultados de la validación cruzada mediante K-folds",
  ),
)

El resultado obtenido es es siguiente:


`Cross-Validation SMAPE Loss: 1.1076 ± 0.0171`

Esto indica que, tras aplicar validación cruzada, el modelo obtuvo un SMAPE medio de 1.1076 (equivalente
a un 110.76 %), con una variabilidad entre particiones de ±0.0171 (≈ 1.71 %). Este valor alto muestra
que las predicciones, en promedio, difieren de los valores reales en más de su magnitud, y la baja desviación
indica que el modelo mantiene un comportamiento similar entre conjuntos de validación. En conclusión, el error es
muy significativo.

== Visualización de los resultados del modelo

Para visualizar las predicciones necesitamos evaluar el modelo para uno de los
casos en concreto y se compararán con los datos reales. Se selecciona un caso
del conjunto de validación.

#align(
  center,
  figure(
    ```python
    i = 0
    real = no_na.iloc[val_idx].to_dict(orient="records")[i]
    def predict(real):
        return dict(
            zip(
                target_cols,
                map(
                    float,
                    model.call(
                        inputs=np.array(
                          [
                            [real[col] for col in feature_cols]
                          ]
                        )
                    ).numpy()[0],
                ),
            )
        ) | {
          k: v for k, v in real.items()
            if k in {f"updrs_{i}" for i in range(1, 5)}
        }

    def to_ys(data):
        return [
            [
                data[
                  f"updrs_{i}{'_plus_' + str(month) if month > 0 else ''}"
                ]
                for month in [0, 6, 12, 24]
            ]
            for i in range(1, 5)
        ]
    ```,
    caption: "Evaluación del modelo para un caso del conjunto de validación y se le da un formato más práctico para graficarlo.",
  ),
)

#align(
  center,
  figure(
    ```python
    def plot(real, colors = ["#61bbb6", "#c3f2f0","#ad56cd", "#4a3b85"]):
        predicted = predict(real)
        x = [0, 6, 12, 24]
        real_ys = to_ys(real)
        predicted_ys = to_ys(predicted)
        for real_y, predicted_y, color, i in zip(
                  real_ys,
                  predicted_ys,
                  colors,
                  range(1,5)
            ):
            plt.plot(x, real_y, color = color, label = f"Real updrs_{i}")
            plt.plot(
              x, predicted_y, '-.', color = color,
              label = f"Predicted updrs_{i}"
            )
        plt.legend()
        plt.gcf().set_size_inches((18,6))
        plt.show()
    ```,
    caption: "Graficar datos reales y prediciones para su comparación.",
  ),
)

Estos son algunos de los resultados. Cada color representa una de las categorías UPDRS. Las líneas continuas representan
los datos reales y las discontinuas las predicciones.

En los ejemplos seleccionados al azar todos los datos de UPDRS-4 figuran como 0. Es muy probable que estos ceros provengan
de los valores faltantes que fueron rellenados. El modelo no predice 0 en ninguno de los ejemplos para UPDRS-4 lo cual puede
que haya contribudo al error.

Para cuantificar el efecto de la categoría 4 de UPDRS, (un caso especial por su grna número de valores faltantes) la eliminamos
tanto de las características de entrada como de las variables objetivos y obtemos un error reducido aunque aún muy significativo:
0.8993 ± 0.0294.

#align(center)[
  #figure(
    [
      #image("../assets/nn-model-results-1.png", width: 100%)
      #image("../assets/nn-model-results-2.png", width: 100%)
      #image("../assets/nn-model-results-3.png", width: 100%)
      #image("../assets/nn-model-results-4.png", width: 100%)
    ],
    caption: [Gráficos comparativos de los datos reales con las predicciones del modelo.],
  )
]

== Incluyendo datos de péptidos

Probamos también a añadir información sobre péptidos y proteínas. Se muestra el proceso para los péptidos, necesitamos
que la información de cada péptido este en una columna distinta, asociada a una visita concreta. Para ello utilizamos una
operación llamada `pivot`.

La operación pivot consiste en reorganizar una tabla para cambiarla de un formato “largo” a un formato
“ancho”. En el formato largo, cada fila suele representar una observación con una etiqueta que indica
de qué categoría es y un valor asociado. El pivot transforma esas etiquetas en columnas separadas, colocando
en cada celda el valor que corresponda.

Este cambio de forma facilita ciertos análisis y comparaciones, ya que cada columna pasa a representar
una categoría específica y cada fila mantiene la identificación única de la observación. Además, si en
la tabla original había varias filas que correspondían a la misma combinación de identificadores y categoría,
se pueden agregar sus valores usando una función como la suma o el promedio, evitando duplicados y
dejando una estructura limpia y consistente. En este caso sabemos que no hay duplicados, pero es necesario que
especifiquemos una función de agregación, especificamos la suma de los valores pero no es relevante.

Por otra parte, recordamos que en la primera sección de este capítulo limpiamos ya los datos de los péptidos
y proteínas normalizando usando el rango.

#align(
  center,
  figure(
    ```python
    with_leads = (
        with_leads.set_index(["patient_id", "visit_month"])
        .join(
            scaled_peptide.pivot_table(
                values="PeptideAbundance",
                index=["patient_id", "visit_month"],
                columns=["Peptide"],
                aggfunc="sum",
            ).fillna(0)
        )
        .reset_index()
    )
    ```,
    caption: "Añadir columnas con los datos de abundancia de péptidos a cada visita.",
  ),
)

Finalmente especifiacmos estas nuevas columnas como características de entrada, para compesar este aumento de tamaño en
los datos de entrada aumentamos también los tamaños de las capas ocultas de la red. Los resultados son peores que cuando
no se incluye esta información: 1.3317 ± 0.0219.

