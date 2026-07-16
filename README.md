# Proyecto 2: Clasificador de imagenes de Tom y Jerry usando deep learning

**Integrantes:** Fernando Palma y Isidora Salgado

## Descripción del problema
El principal objetivo de este proyecto es usar un modelo Deep Learning pre-entrenado para que sea capaz de clasificar imagenes de la serie animada `Tom & Jerry` aplicando los conocimientos aprendidos en la unidad 3 del curso.

## Descripción del dataset
El dataset tiene por nombre `Tom and Jerry Image Classification`, el cual proviene de la pagina de [Kaggle](https://www.kaggle.com/datasets/balabaskar/tom-and-jerry-image-classification?resource=download). 

El dataset contiene exactamente 5478 imagenes extraidas de capitulos de la serie animada, en donde cada imagen corresponde a un frame del capitulo.
El etiquetado de las imagenes se realizo de manera manual, por lo que la precision de las etiquetas de referencia es del 100%.

Las imagenes se encuentran estructuradas en los siguientes directorios:
1) **`tom:`** contiene imagenes exclusivamente con el personaje Tom.
2) **`jerry:`** contiene imagenes exclusivamente con el personaje Jerry.
3) **`tom_jerry_1:`** contiene imagenes en las cuales aparecen ambos personajes, Tom y Jerry.
4) **`tom_jerry_0:`** contiene imagenes de fondo o sotuaciones en las cuales no aparecen ninguno de los dos personajes.

Ademas, hay tambien un archivo `ground_truth.csv` que contiene el mapeo exacto de cada imagen con su etiqueta correspondiente

## Modelo seleccionado y justificación
Para llevar a cabo la implementación de este problema de clasificación, se implementó la arquitectura **ResNet50** utilizando `PyTorch` inicializada con los pesos pre-entrenados del conjunto de datos `ImageNet`.

La elección de este modelo se basa principalmente en:
* La extracción de caracteristicas, ya que dado el formato de las imagenes, trazos simples y colores planos, se opto por aprovechar los filtros optimizados por ResNet50 para la deteccion de bordes y texturas.
* La estabilidad del gradiente, ya que una de sus mejores ventajas son sus conexiones residuales las cuales ayudan con el problema del desvanecimiento del gradiente.

## Metodología
