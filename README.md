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
Se aplicaron transformaciones con `torchvision.transforms` para adecuar las imágenes al formato requerido por `ResNet50`. Se redimensionaron a 256x256 píxeles y se les aplicó un recorte central de 224x224. Al set de entrenamiento se le agregó volteo horizontal aleatorio y rotaciones de hasta 15°, para reducir overfitting dado el desbalance entre clases. Finalmente, las imágenes se convirtieron en tensores y se normalizaron con la media y desviación estándar de ImageNet, requisito del transfer learning.

```python
import torchvision.transforms as transforms

transformaciones_train = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transformaciones_eval = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

### 3. Estructuración de datos

Las 5.478 imágenes totales se dividieron de forma reproducible (`seed=1`) en 70% entrenamiento (3.834 imágenes), 15% validación (821 imágenes) y 15% test (823 imágenes), generando los índices con `torch.randperm` y repartiéndolos mediante `Subset` sobre dos instancias de `ImageFolder`, una con transformaciones de entrenamiento y otra con las de evaluación, para que cada índice reciba el preprocesamiento correcto según su conjunto. Luego se usaron `DataLoaders` para iterar en lotes de 32 imágenes, mezclando solo el conjunto de entrenamiento para mejorar la generalización, mientras que validación y test se mantienen en orden fijo al no participar en la actualización de pesos.

```python
dataset_train_full = ImageFolder('./imagenes', transform=transformaciones_train)
dataset_eval_full = ImageFolder('./imagenes', transform=transformaciones_eval)

n_total = len(dataset_train_full)
entrenamiento_tam = int(0.70 * n_total)
validacion_tam = int(0.15 * n_total)
test_tam = n_total - entrenamiento_tam - validacion_tam

generador = torch.Generator().manual_seed(1)

indices = torch.randperm(n_total, generator=generador).tolist()
train_indices = indices[:entrenamiento_tam]
val_indices = indices[entrenamiento_tam:entrenamiento_tam + validacion_tam]
test_indices = indices[entrenamiento_tam + validacion_tam:]

train_dataset = Subset(dataset_train_full, train_indices)
val_dataset = Subset(dataset_eval_full, val_indices)
test_dataset = Subset(dataset_eval_full, test_indices)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"Clases detectadas automáticamente: {dataset_train_full.classes}")
print(f"Total de imágenes para entrenamiento: {len(train_dataset)}")
print(f"Total de imágenes para validación: {len(val_dataset)}")
print(f"Total de imágenes para test: {len(test_dataset)}")
```

```
Clases detectadas automáticamente: ['jerry', 'tom', 'tom_jerry_0', 'tom_jerry_1']
Total de imágenes para entrenamiento: 3834
Total de imágenes para validación: 821
Total de imágenes para test: 823
```

Dado el desbalance identificado en el EDA (`tom` predomina sobre `tom_jerry_1`), se calcularon pesos por clase inversamente proporcionales a su frecuencia en el conjunto de entrenamiento. Estos pesos se incorporaron en la función de pérdida (`CrossEntropyLoss`) para que el modelo penalice con mayor fuerza los errores en las clases con menos imágenes, evitando que favorezca a la clase mayoritaria.

```
jerry:       1.098
tom:         0.702
tom_jerry_0: 0.905
tom_jerry_1: 1.788
```

## Entrenamiento (Optimización)

Se cargó la arquitectura `ResNet50` preentrenada en ImageNet, congelando los parámetros de sus capas convolucionales para conservar el conocimiento del preentrenamiento, y se reemplazó la última capa por una con 4 salidas, una por cada categoría (`jerry`, `tom`, `tom_jerry_0`, `tom_jerry_1`). El modelo se entrenó durante 20 `epochs` usando `CrossEntropyLoss`, aplicado únicamente sobre los parámetros de la capa `fc`. Al finalizar cada época, el modelo se evaluó sobre el conjunto de validación sin actualizar sus parámetros.

Los resultados del entrenamiento por época fueron los siguientes:

![Epochs](/graficos/epochs.png)

La pérdida de entrenamiento baja fuerte en las primeras épocas y luego se estabiliza recién en el último tercio, cerca de las épocas 12-13, cuando entra al rango 0.60-0.65, cerrando en 0.6138. La precisión de entrenamiento sube de 53.83% a un máximo de 76.94% en la época 19, para terminar en 75.53% en la época 20. La validación se mueve harto más entre época y época, lo que tiene sentido porque el conjunto de validación es chico y unos pocos errores por lote pueden mover bastante el promedio.

También es posible observar esto en el gráfico de curvas. El gráfico izquierdo muestra la evolución de la pérdida (`loss`) y el derecho la precisión (`accuracy`), comparando entrenamiento y validación a lo largo de las 20 épocas:

![Curvas de entrenamiento](/graficos/curvas_entrenamiento.png)

La curva de `Train Loss` cae fuerte en las primeras 5 épocas y después se mueve dentro de un rango acotado sin retrocesos importantes. La de `Val Loss` tiene dos picos claros entre las épocas 7 y 9, y después de eso baja y se queda bajo 0.65 hasta el final. Con **accuracy**, `Train Accuracy` sube de forma sostenida durante todo el entrenamiento, mientras que `Val Accuracy` sube rápido al principio y después oscila entre 70% y 78%, terminando cerca de 77%. La diferencia de estabilidad entre ambas curvas va de la mano con el tamaño chico del set de validación, y no apunta a un sobreajuste severo porque las dos curvas de pérdida terminan en valores parecidos.

## Control de Overfitting

Para controlar el overfitting se combinaron dos estrategias: **data augmentation** en el conjunto de entrenamiento, que amplía la variedad visual y reduce la sensibilidad del modelo a la orientación de los personajes; y **pesos por clase** en la `CrossEntropyLoss`, que penalizan más los errores en clases minoritarias, evitando que el modelo las ignore.

El comportamiento observado confirma que estas medidas funcionaron, la pérdida de validación no muestra un aumento sostenido respecto a la de entrenamiento, y la separación final entre ambas (0.6138 vs 0.6363) es moderada.

## Evaluación en el Conjunto de Test

Una vez finalizado el entrenamiento, el modelo fue evaluado en el conjunto de test (823 imágenes), el cual no fue utilizado en ningún momento durante el entrenamiento ni la validación.

```
Accuracy en test: 0.7485
```

El modelo obtuvo una `accuracy` del 74.85% en el conjunto de test, resultado coherente con las métricas observadas durante la validación hacia el final del entrenamiento (~77%).

## Métricas de Desempeño

Ahora, el reporte de clasificación por clase sobre el conjunto de test fue el siguiente:

|              | Precision | Recall | F1-Score | Support |
|--------------|-----------|--------|----------|---------|
| **jerry**       | 0.799     | 0.733  | 0.765    | 195     |
| **tom**         | 0.793     | 0.749  | 0.770    | 271     |
| **tom_jerry_0** | 0.705     | 0.835  | 0.765    | 243     |
| **tom_jerry_1** | 0.670     | 0.588  | 0.626    | 114     |
| **accuracy**    |           |        | **0.748**| 823     |
| **macro avg**   | 0.742     | 0.726  | 0.731    | 823     |
| **weighted avg**| 0.751     | 0.748  | 0.747    | 823     |

Analizando los resultados por clase, `jerry`, `tom` y `tom_jerry_0` obtienen F1-scores similares, entre 0.765 y 0.770, mostrando un desempeño equilibrado entre precisión y recall. La clase `tom_jerry_0` (escenas sin personajes) presenta el recall más alto de todas (0.835), lo que indica que el modelo identifica con bastante consistencia las escenas de fondo, aunque a costa de una precisión más baja (0.705), producto de que probablemente absorbe algunas confusiones de otras clases. 

En el extremo opuesto, `tom_jerry_1` vuelve a ser la clase más difícil, con el F1-score más bajo (0.626) y un recall de apenas 0.588, el modelo pierde cerca del 40% de las imágenes donde aparecen ambos personajes. Esto ocurre tanto por su menor representación en el dataset (peso 1.788 en la función de pérdida) como por la dificultad visual del caso, cuando aparecen dos personajes en la misma escena suele haber superposiciones y deformaciones propias del estilo caricaturesco, lo que complica distinguir los rasgos de cada uno frente a las clases de un solo personaje.

Es importante destacar que las cuatro categorías comparten el mismo estilo visual y fondo animado, lo que dificulta la separación entre clases y ayuda a explicar por qué el accuracy general se queda en 74.85% en vez de ser más alto, considerando además que solo se entrenó la última capa del modelo.

## Predicción en Imágenes Nuevas

Se seleccionaron aleatoriamente dos tandas de 5 imágenes del conjunto de test para hacer una validación cualitativa, mostrando cada imagen junto a su clase real, la clase predicha por el modelo y el porcentaje de confianza asociado. Esta revisión permite ver de forma directa si el modelo identifica correctamente a los personajes en distintas poses, planos y contextos de la animación.

![Resultados 1](/graficos/resultado1.png)

![Resultados 2](/graficos/resultado2.png)

De las 10 imágenes evaluadas, el modelo acertó en 8, un resultado algo superior al accuracy general del test, aunque dentro de un margen esperable tratándose de una muestra pequeña. Los dos errores comparten un mismo patrón: en ambos casos, `tom` predicho como `tom_jerry_0` y `jerry` predicho como `tom_jerry_0`, el personaje real aparece pequeño y distante en el encuadre, con el fondo ocupando la mayor parte de la imagen. Esto sugiere que el modelo se apoya bastante en el contexto general de la escena, y cuando el personaje no domina el cuadro, tiende a confundirlo con `tom_jerry_0`, la clase de escenas de fondo. Este comportamiento coincide con lo observado en el reporte de clasificación, donde `tom_jerry_0` tuvo el recall más alto (0.835): el modelo tiende a asignar a esa clase los casos que le resultan ambiguos.

También hay dos aciertos con confianza baja, 56.5% y 49.8%, cercanos al punto donde el modelo duda entre clases. El segundo corresponde justamente a `tom_jerry_1`, la clase más débil según las métricas (F1-score 0.626), lo que confirma que el modelo la predice bien en algunos casos, pero con menos seguridad que en el resto.

## Conclusión general

Clasificar fotogramas de Tom y Jerry en 4 categorías resultó más exigente que separar clases visualmente distintas, porque los cuatro grupos comparten el mismo estilo de dibujo, la misma paleta de colores y fondos que se repiten entre escenas. Eso explica en gran parte el 74.85% de accuracy obtenido en el test, un resultado sólido pero lejos de lo que se logra en problemas donde las categorías se diferencian a simple vista.

El desempeño no fue uniforme entre clases, y esa diferencia cuenta algo relevante sobre el problema. `jerry`, `tom` y `tom_jerry_0` alcanzaron F1-scores cercanos a 0.77, mientras que `tom_jerry_1` se quedó en 0.626, con un recall de apenas 0.588. La causa no fue solo la menor cantidad de ejemplos de esa clase en el entrenamiento, ya que se aplicaron pesos por clase justamente para compensar ese desbalance. Gran parte del error se debe a la dificultad visual propia de la categoría: cuando ambos personajes comparten pantalla, suele haber superposición entre ellos y menos espacio limpio para que el modelo distinga rasgos característicos de cada uno.

El modelo cumplió el objetivo del proyecto y el resultado va acorde a la complejidad del dataset y a las decisiones tomadas durante su construcción, en particular haber entrenado solo la capa fc sobre un modelo congelado.

## Declaración de uso de IA

Se declara que el desarrollo de este proyecto se basó principalmente en los contenidos y ejemplos revisados en clases durante la unidad 3 del curso. Se utilizó apoyo de inteligencia artificial para resolver dudas puntuales de implementación en `PyTorch`, redactar y ordenar partes de este documento, y revisar la coherencia del código, manteniendo siempre las clases como guía principal para las decisiones metodológicas del proyecto.