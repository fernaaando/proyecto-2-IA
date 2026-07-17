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
4) **`tom_jerry_0:`** contiene imagenes de fondo o situaciones en las cuales no aparecen ninguno de los dos personajes.

Ademas, hay tambien un archivo `ground_truth.csv` que contiene el mapeo exacto de cada imagen con su etiqueta correspondiente

## Modelo seleccionado y justificación
Para llevar a cabo la implementación de este problema de clasificación, se implementó la arquitectura **ResNet50** utilizando `PyTorch` inicializada con los pesos pre-entrenados del conjunto de datos `ImageNet`.

La elección de este modelo se basa principalmente en:
* La extracción de caracteristicas, ya que dado el formato de las imagenes, trazos simples y colores planos, se opto por aprovechar los filtros optimizados por ResNet50 para la deteccion de bordes y texturas.
* La estabilidad del gradiente, ya que una de sus mejores ventajas son sus conexiones residuales las cuales ayudan con el problema del desvanecimiento del gradiente.

## Metodología

### 1. Analisis Exploratorio de Datos (EDA)
Para comprender la estructura y formato del dataset original con el que se trabajara, se realizo un analisis exploratorio antes de la construccion y entrenamiento del modelo:

* **Transformación de etiquetas:** se proceso el archivo `ground_truth.csv`en el cual se presentaba una codificacion binaria de cada personaje y luego se aplico una funcion `asignar_clase` para unificar las etiquetas en 4 clases categoricas (`tom`, `jerry `, `tom_jerry_0`, `tom_jerry_1`)

* **Analisis de Distribución:** mediante graficos de barras, se identifico un desbalance de clases en el dataset, ya que predomina la clase `tom` y hay una menor presencia de la clase donde aparecen ambos personajes `tom_jerry_1`. Esto justifico la necesidad de implementar tecnicas de regularizacion en etapas posteriores.

![Distribucion de clases](/graficos/distribucion_clases.png)

* **Analisis dimensional:** se extrajeron y graficaron las resoluciones de las 5.478 imagenes. El diagrama de dispersion mostro que los fotogramas provenian de resoluciones estandarizadas, lo que requirio de la necesidad de aplicar transformaciones geometricas para adaptar los tensores a la arquitectura de `ResNet50`.

![Distribucion de dimensiones](/graficos/grafico_dimensiones.png)

### 2. Feature Engineering
Luego de realizar el EDA, se realizaron transformaciones a las imagenes utilizando la libreria `torchvision.transforms`, esto con la finalidad de tener el formato que sea funcional con ResNet50:

* **Estandarización Geometrica:** se redimensionaron todas las imagenesa 256x256 pixeles y luego se realizó un recorte central (`CenterCrop`) de 224x224 pixeles para cumplir con los requisitos de dimensionalidad que tiene `ResNet50`

* **Control de overfitting:** para evitar que el modelo memorice las imagenes provocado en cierta parte por el desbalance de las imagenes de cada clase y la naturaleza de estas, se aplico ciertas medidas para mitigar el sobreajuste en el conjunto de entrenamiento; en primera instancia se incluyo un volteo horizontal de cada imagen de forma aleatoria con un 50% de probabilidades de tener esta transformacion (`RandomHorizontalFlip`) y en segunda medida se realizaron rotaciones de hasta 15° a ciertas imagenes (`RandomRotation`)

* **Normalización Matematica:** finalmente, las imagenes se convirtieron en Tensores de PyTorch y ademas, sus canales RGB fueron normalizados utilizando la media y desviacion estandar del dataset ImageNet (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`), para asi garantizar la compatibilidad matematica para el `transfer learning`

```python
import torchvision.transforms as transforms

transformaciones = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

### 3. Estructuración de datos

**División del Dataset:** se tomaron las 5.478 imagenes totales ya procesadas con todas las transformaciones anteriores y luego se dividieron de forma reproducible, usando una seed fija (1), en dos subconjuntos: 80% para el entrenamiento del modelo (4.382 imagenes) y 20% para la validacion del modelo (1.096 imagenes)

**Empaquetado eficiente:** se usaron `DataLoaders`de PyTorch para iterar sobre las imagenes en lotes de 32 imagen cada uno, luego, se aplico un ordenamiento aleatorio (`shuffle=True`) solo al conjunto de entrenamiento para mejorar la capacidad de generalizacion del modelo al momento de entrenarse

```python
from torch.utils.data import DataLoader, random_split
import torch

#semilla (1)
generador = torch.Generator().manual_seed(1)
train_dataset, val_dataset = random_split(dataset_completo, [entrenamiento_tam, validacion_tam], generator=generador)

#dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
```