= Conclusiones

== Resumen de resultados
El objetivo principal de este trabajo consistía en evaluar el potencial de los modelos Seq2Seq en la
predicción de la progresión del Parkinson mediante la escala UPDRS, explorando si esta arquitectura podría
superar las limitaciones de enfoques más tradicionales en el análisis de datos temporales biomédicos.

Los resultados obtenidos demuestran que el modelo Seq2Seq logró un rendimiento significativamente superior
al modelo de referencia basado en redes neuronales simples. Específicamente, se alcanzó una mejora del
30-40% en la métrica SMAPE, reduciendo el error de validación hasta aproximadamente un 75%. Esta mejora
sustancial confirma la capacidad del modelo para capturar dependencias temporales complejas en la progresión
de la enfermedad que los modelos más simples no logran explotar.

El análisis de diferentes configuraciones reveló que el modelo base (utilizando únicamente mes de visita
y valores UPDRS previos) proporcionó los mejores resultados, manteniéndose consistente tanto con arquitecturas
GRU como LSTM. Sin embargo, los intentos de incorporar información adicional proveniente de datos proteómicos
y peptídicos no resultaron en mejoras significativas del rendimiento, manteniendo el error en torno al mismo nivel del 75%.

Si bien estos resultados representan un avance conceptual importante y demuestran la validez del enfoque
Seq2Seq para este tipo de problemas, el nivel de error obtenido aún no permite considerar el modelo como
clínicamente aplicable de forma robusta. No obstante, la capacidad demostrada para capturar tendencias
en las distintas categorías UPDRS sugiere que la arquitectura tiene potencial para futuras mejoras con
datasets más amplios y completos.

== Limitaciones

La limitación principal que condicionó el desarrollo y los resultados de este trabajo fue la cantidad
limitada de datos disponibles. Con aproximadamente 2.600 visitas que incluían datos de proteínas y 5.800
con información UPDRS, el volumen de datos resultó insuficiente para explotar completamente el potencial
de los modelos de deep learning, que típicamente requieren grandes cantidades de información para generalizar
adecuadamente.

Un problema adicional significativo fue la alta proporción de valores faltantes, especialmente pronunciada
en la categoría UPDRS IV, lo que limitó la capacidad del modelo para aprender patrones completos y consistentes.
Esta fragmentación en los datos se vio agravada por la heterogeneidad en las mediciones y la escasez
de seguimientos longitudinales consistentes para los mismos pacientes a lo largo del tiempo.

Como se detalla en los capítulos previos dedicados al análisis exploratorio y preprocesamiento de datos,
estas limitaciones no son inherentes al modelo propuesto, sino que derivan directamente de las características
y restricciones de la fuente de datos disponible. El conjunto de datos, pese a ser una iniciativa valios
, presenta las limitaciones típicas de los estudios clínicos reales: alta variabilidad en el seguimiento
de pacientes, protocolos de medición heterogéneos y dificultades en la recolección sistemática de datos longitudinales.

La tabla de resultados presentada evidencia que las variaciones en hiperparámetros
(tamaño oculto, número de capas, tasa de dropout) no produjeron mejoras sustanciales, lo que sugiere
que el modelo alcanzó rápidamente el límite de información extraíble del dataset disponible. Esta saturación
temprana es característica de escenarios con datos limitados, donde aumentar la complejidad del modelo
no aporta beneficios adicionales.

== Futuras líneas de trabajo

=== Expansión del conjunto de datos

La ampliación significativa del dataset representa la línea de trabajo más crítica para mejorar sustancialmente
el rendimiento del modelo. Esta expansión podría abordarse desde múltiples frentes:
La integración de datos abiertos anonimizados (Open Data) procedentes de otras iniciativas de investigación
similares permitiría aumentar considerablemente el volumen de observaciones longitudinales disponibles

El desarrollo de una interfaz web colaborativa constituye otra vía prometedora para la recolección sistemática
de datos clínicos y demográficos. Esta plataforma podría facilitar la contribución de centros médicos
especializados, permitiendo la estandarización de protocolos de medición y el seguimiento más consistente
de cohortes de pacientes.

La inclusión de variables adicionales como edad, sexo, hábitos de vida, comorbilidades y factores socioeconómicos
podría enriquecer significativamente el poder predictivo del modelo. Es evidente que estos factores contextuales
tienen una influencia relevante en la progresión de enfermedades neurodegenerativas.

=== Optimización de hiperparámetros
El enfoque utilizado en este trabajo para la selección de hiperparámetros se basó principalmente en prueba
y error guiada por conocimiento del dominio. Futuras iteraciones deberían incorporar métodos automáticos
de búsqueda de hiperparámetros que permitan explorar de forma más sistemática y eficiente el espacio
de configuraciones posibles.
Técnicas como optimización bayesiana, búsqueda en malla (grid search) o búsqueda aleatoria (random search)
podrían identificar configuraciones más óptimas que las encontradas manualmente. Adicionalmente, el uso
de plataformas de *AutoML* podría automatizar no solo la búsqueda de hiperparámetros sino también la selección
de arquitecturas de modelo más apropiadas para este dominio específico.

=== Extensión del marco experimental

El _transfer learning_ con modelos preentrenados representa una oportunidad particularmente relevante.
Modelos entrenados en datasets médicos más amplios o en tareas relacionadas de análisis de series temporales
podrían transferir conocimientos útiles que compensen la limitación de datos específicos del Parkinson.

La exploración de arquitecturas híbridas que combinen efectivamente datos clínicos, proteómicos y demográficos
mediante diferentes modalidades de entrada podría superar las limitaciones observadas en este trabajo.
Técnicas de fusión multimodal tardía o temprana podrían permitir que cada tipo de información contribuya
de forma más efectiva al modelo final. Con multimodalidad nos referimos aquí, a integrar datos numéricos con,
por ejemplo, imágenes de escáneres cerebrales.

== Valor del aprendizaje realizado

Desde la perspectiva académica, este proyecto representa una aplicación integral de técnicas avanzadas
de Deep Learning a un problema real de impacto social significativo, cumpliendo plenamente los objetivos
formativos de un Trabajo de Fin de Grado. El desarrollo completo del ciclo de un proyecto de Machine
Learning, desde el análisis exploratorio inicial hasta la evaluación final de resultados, ha permitido
adquirir experiencia práctica en cada una de las etapas críticas: preprocesamiento de datos complejos,
diseño de arquitecturas de modelo, implementación técnica y análisis crítico de resultados.

Más allá de los aspectos técnicos específicos, este trabajo posee un impacto conceptual relevante dentro
del contexto actual del desarrollo de la inteligencia artificial. El modelo Seq2Seq utilizado constituye
una de las arquitecturas precursoras directas de los *Transformers*, que han revolucionado el campo y
dado lugar al auge actual de los *Grandes Modelos de Lenguaje* (_LLM_). Haber trabajado en profundidad
con mecanismos de atención, codificadores-decodificadores y el procesamiento de secuencias temporales
proporciona una base sólida para comprender arquitecturas modernas como BERT, GPT o T5, que dominan el
panorama actual de la inteligencia artificial.

La experiencia adquirida trasciende los resultados numéricos específicos obtenidos. El proceso de enfrentarse
a las limitaciones reales de los datos, la necesidad de tomar decisiones metodológicas fundamentadas
ante la incertidumbre, y la interpretación crítica de resultados en un contexto biomédico, representan
aprendizajes fundamentales para el desarrollo profesional en el campo del Machine Learning aplicado.

== Conclusión final

Este trabajo demuestra que los modelos Seq2Seq poseen potencial significativo para
la predicción de la progresión del Parkinson, logrando mejoras sustanciales respecto a enfoques más simples.
Aunque las limitaciones del dataset impidieron alcanzar niveles de precisión clínicamente aplicables
los fundamentos metodológicos desarrollados y las líneas de trabajo futuras identificadas proporcionan
una base sólida para investigaciones posteriores que, con mayores volúmenes de datos, podrían contribuir
efectivamente al diagnóstico y seguimiento clínico de esta enfermedad neurodegenerativa.

