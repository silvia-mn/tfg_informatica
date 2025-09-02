#import "uva-template/template.typ": template
#import "@preview/zebraw:0.5.5": *
#show: zebraw

#show: template.with(
  title: [Predicción del avance de la enfermedad de Parkinson mediante técnicas de _Machine Learning_],
  author: "Silvia Muñoz Nogales",
  female_author: true,
  tutor: "José Vicente Álvarez Bravo",
  bibliography_path: "/bibliografia.bib",
)

#for file in (
  "chapters/introduccion.typ",
  "chapters/planificacion.typ",
  "chapters/eda.typ",
  "chapters/nn.typ",
  "chapters/solucion_propuesta.typ",
  "chapters/impl-seq2seq.typ",
  "chapters/conclusiones.typ",
) [
  #include (file)
]
