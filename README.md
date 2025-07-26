# Modelo Entre Diagonales - Clasificador de Imágenes

## 📋 Descripción

Este proyecto implementa un sistema de clasificación de imágenes utilizando modelos de deep learning preentrenados. Permite entrenar y evaluar diferentes arquitecturas de redes neuronales convolucionales para tareas de clasificación de imágenes.

## 🏗️ Estructura del Proyecto

```
modelo-entre-diagonales/
├── config.py           # Configuraciones globales del proyecto
├── models.py           # Definiciones de arquitecturas de modelos
├── train.py            # Script principal para entrenamiento
├── predict.py          # Script para realizar predicciones
├── utils.py            # Funciones utilitarias y auxiliares
├── requirements.txt    # Dependencias del proyecto
├── docs/               # Documentación técnica detallada
│   ├── README.md       # Índice de documentación
│   ├── arquitecturas.md        # Explicación de modelos
│   ├── funciones_activacion.md # Conceptos de activación
│   ├── keras_tensorflow.md     # Guía de Keras/TensorFlow
│   ├── variables_configuracion.md # Variables del proyecto
│   ├── metricas_evaluacion.md  # Sistema de evaluación
│   └── preprocesamiento.md     # Procesamiento de imágenes
├── data/               # Directorio de datos (estructura requerida)
│   ├── clase1/
│   │   ├── train/
│   │   └── test/
│   └── clase2/
│       ├── train/
│       └── test/
├── models/             # Modelos entrenados (generado automáticamente)
├── plots/              # Gráficas de entrenamiento (generado automáticamente)
├── experiments/        # Historial de experimentos (generado automáticamente)
└── temp_dataset/       # Directorio temporal (generado automáticamente)
```

## 🚀 Instalación

### Prerrequisitos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)

### Dependencias

Instala las siguientes librerías:

```bash
pip install tensorflow tensorflow-hub
pip install matplotlib seaborn
pip install scikit-learn
pip install pillow
pip install rich
pip install numpy pandas
```

O crea un archivo `requirements.txt` con el siguiente contenido:

```txt
tensorflow>=2.10.0
tensorflow-hub>=0.12.0
matplotlib>=3.5.0
seaborn>=0.11.0
scikit-learn>=1.1.0
pillow>=9.0.0
rich>=12.0.0
numpy>=1.21.0
pandas>=1.4.0
```

Y ejecuta:

```bash
pip install -r requirements.txt
```

## 📊 Preparación de Datos

### Estructura Requerida

Los datos deben organizarse en la siguiente estructura:

```
data/
├── clase1/
│   ├── train/
│   │   ├── imagen1.jpg
│   │   ├── imagen2.jpg
│   │   └── ...
│   └── test/
│       ├── imagen1.jpg
│       ├── imagen2.jpg
│       └── ...
├── clase2/
│   ├── train/
│   │   └── ...
│   └── test/
│       └── ...
└── ...
```

### Formatos de Imagen Soportados

- JPG/JPEG
- PNG
- JFIF (se convierte automáticamente a JPG)

## 📊 Análisis del Dataset

Antes de comenzar el entrenamiento, es recomendable analizar la distribución y características de tu dataset. El proyecto incluye un script especializado para este propósito.

### Script de Análisis

```bash
# Ejecutar análisis completo del dataset
python analyze_dataset.py
```

### Características del Análisis

El script `analyze_dataset.py` genera:

#### 📋 **Reporte Detallado**
- Estadísticas por clase (número de imágenes, tamaños)
- Distribución train/test por cada clase
- Detección de archivos con extensiones no válidas
- Análisis de balance del dataset

#### 📈 **Visualizaciones Automáticas**
- Gráfico de barras: distribución train vs test por clase
- Gráfico de pastel: distribución total por clase
- Gráfico de balance: ratio train/test por clase

#### 📁 **Archivos Generados**
```
dataset-reports/
├── dataset_report.txt        # Reporte detallado en texto
└── plots/
    ├── dataset_distribution.png
    └── dataset_balance.png
```

### Ejemplo de Uso y Salida

```bash
$ python analyze_dataset.py

Analizando dataset...
============================================================
Clases encontradas: 6
Clases: casa-curutchet, catedral, dardo-rocha, municipalidad, museo-naturales, teatro-argentino

casa-curutchet:
  Train: 30 imágenes (531.0 KB)
  Test:  10 imágenes (177.2 KB)
  Total: 40 imágenes (708.2 KB)

✓ Visualizaciones guardadas en 'dataset-reports/plots/'
✓ Reporte guardado como 'dataset-reports/dataset_report.txt'
```

### Ventajas del Análisis Previo

- **Detecta desbalances** en el dataset antes del entrenamiento
- **Identifica archivos problemáticos** con extensiones no válidas
- **Valida la estructura** requerida del proyecto
- **Proporciona métricas base** para comparar con resultados del modelo

> **💡 Recomendación**: Ejecuta este análisis cada vez que modifiques tu dataset para mantener un control de calidad.

## 🤖 Modelos Disponibles

El proyecto incluye cuatro arquitecturas de modelos preentrenados:

### 1. **EfficientNetB3**
- **Tamaño de imagen**: 300x300 píxeles
- **Características**: Modelo eficiente con excelente balance precisión/velocidad
- **Uso recomendado**: Para datasets complejos con alta variabilidad

### 2. **MobileNetV2**
- **Tamaño de imagen**: 224x224 píxeles
- **Características**: Modelo ligero optimizado para dispositivos móviles
- **Aumentación de datos**: Incluye rotación, zoom y flip horizontal
- **Uso recomendado**: Para aplicaciones con recursos limitados

### 3. **ResNeSt50**
- **Tamaño de imagen**: 224x224 píxeles
- **Características**: Basado en ResNet con mejoras arquitectónicas
- **Uso recomendado**: Para datasets que requieren alta precisión

### 4. **ConvNeXt-Tiny**
- **Tamaño de imagen**: 224x224 píxeles
- **Características**: Arquitectura moderna que combina convolutiones con ideas de Transformers
- **Dropout**: 0.3 para regularización
- **Uso recomendado**: Para datasets modernos que requieren arquitecturas de vanguardia

## 🎯 Entrenamiento

### Comandos Básicos

```bash
# Ver modelos disponibles
python train.py --list-models

# Entrenar con EfficientNet
python train.py --model efficientnet

# Entrenar con MobileNet
python train.py --model mobilenet

# Entrenar con ResNeSt
python train.py --model resnest

# Entrenar con ConvNext
python train.py --model convnext

# Entrenar con número personalizado de épocas
python train.py --model efficientnet --epochs 25
```

### Configuración Avanzada

Puedes modificar los parámetros en `config.py`:

```python
MODEL_SPECIFIC_CONFIG = {
    "mobilenet": {
        "img_size": (244, 244),
        "learning_rate": 0.001,
        "dropout_rate": 0.2,
        "use_augementation": True,
        "dense_units": 64,
    },
    # ... otros modelos
}
```

### Proceso de Entrenamiento

1. **Validación de datos**: Verifica la estructura del directorio
2. **Conversión de formatos**: Convierte archivos JFIF a JPG automáticamente
3. **Preparación de datasets**: Crea estructura temporal optimizada
4. **Entrenamiento**: Ejecuta el entrenamiento con validación
5. **Evaluación**: Genera métricas y visualizaciones
6. **Guardado**: Almacena modelo, metadatos y resultados

## 📈 Resultados y Visualizaciones

Durante el entrenamiento se generan:

### Archivos de Salida

```
models/[modelo]_[timestamp]/
├── model.h5                    # Modelo entrenado
├── metadata.json              # Metadatos del modelo
├── evaluation_per_class.json  # Métricas por clase
└── report.json                # Reporte de clasificación completo

plots/[timestamp]/
├── training_metrics.png       # Curvas de accuracy y loss
└── confusion_matrix.png       # Matriz de confusión

experiments/
└── history.csv                # Historial de todos los experimentos
```

### Métricas Incluidas

- **Accuracy** y **Validation Accuracy**
- **Loss** y **Validation Loss**
- **F1-Score** (macro promedio)
- **Precision**, **Recall** y **F1-Score** por clase
- **Matriz de confusión**

## 🔮 Predicciones

### Uso del Script de Predicción

```bash
# Listar modelos disponibles
python predict.py --list-models

# Predecir imagen individual
python predict.py --model [nombre_modelo] --image ruta/a/imagen.jpg

# Predecir múltiples imágenes
python predict.py --model [nombre_modelo] --batch ruta/a/directorio/

# Mostrar probabilidades detalladas
python predict.py --model [nombre_modelo] --image imagen.jpg --show-probs
```

### Ejemplo de Salida

```
🎯 Predicción para imagen.jpg:
   Clase predicha: clase1 (95.67%)
   
📊 Probabilidades por clase:
   clase1: 95.67%
   clase2: 3.21%
   clase3: 1.12%
```

## ⚙️ Configuración del Proyecto

### Archivos de Configuración

- **`config.py`**: Configuraciones globales y parámetros por modelo
- **`models.py`**: Definiciones de arquitecturas y hiperparámetros
- **`utils.py`**: Funciones auxiliares y utilidades

### Directorios Principales

- **`data/`**: Datasets de entrenamiento y test
- **`models/`**: Modelos entrenados con timestamps
- **`plots/`**: Visualizaciones y gráficas
- **`experiments/`**: Historial de experimentos en CSV

## 🐛 Solución de Problemas

### Errores Comunes

1. **"Directorio de datos no válido"**
   - Verifica que la estructura de carpetas sea correcta
   - Asegúrate de tener subcarpetas `train/` y `test/` para cada clase

2. **"Modelo no encontrado"**
   - Verifica que el modelo especificado esté en la lista de modelos disponibles
   - Usa `python train.py --list-models` para ver opciones

3. **"Error de memoria"**
   - Reduce el `batch_size` en la configuración del modelo
   - Considera usar un modelo más ligero como MobileNet

4. **"Dependencias faltantes"**
   - Instala todas las librerías requeridas
   - Verifica la versión de TensorFlow (≥2.10.0)

### Logging y Depuración

El proyecto utiliza `rich` para logging colorido. Los mensajes incluyen:
- ✅ Operaciones exitosas
- ⚠️ Advertencias
- ❌ Errores
- 📊 Información de progreso

## 🎨 Personalización

### Agregar Nuevos Modelos

1. Define la función de creación en `models.py`:
```python
def create_mi_modelo(img_size, num_classes):
    # Implementación del modelo
    return model, preprocess_fn, "mi_modelo"
```

2. Agrega la configuración en `MODEL_CONFIGS`:
```python
"mi_modelo": {
    "img_size": (224, 224),
    "batch_size": 32,
    "epochs": 15,
    "create_fn": create_mi_modelo,
    "description": "Descripción del modelo",
}
```

### Modificar Aumentación de Datos

Edita las funciones de modelo en `models.py` para agregar o modificar aumentaciones:

```python
augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomBrightness(0.1),  # Nueva aumentación
])
```

## 📚 Documentación Técnica

Para información detallada sobre conceptos específicos, consulta la carpeta [`docs/`](docs/):

### Documentos Disponibles:
- **[Índice General](docs/README.md)** - Navegación completa de la documentación
- **[Arquitecturas de Modelos](docs/arquitecturas.md)** - EfficientNet, MobileNet, ResNeSt
- **[Funciones de Activación](docs/funciones_activacion.md)** - ReLU, Softmax, conceptos matemáticos
- **[Keras y TensorFlow](docs/keras_tensorflow.md)** - Guía completa de frameworks
- **[Variables y Configuraciones](docs/variables_configuracion.md)** - Parámetros del proyecto
- **[Métricas y Evaluación](docs/metricas_evaluacion.md)** - Accuracy, Loss, F1-Score, matrices
- **[Preprocesamiento](docs/preprocesamiento.md)** - Normalización, aumentación, pipelines

### Guías Rápidas:
- **Principiantes**: Empieza con [Keras/TensorFlow](docs/keras_tensorflow.md) y [Funciones de Activación](docs/funciones_activacion.md)
- **Configuración**: Revisa [Variables](docs/variables_configuracion.md) para personalizar el proyecto
- **Problemas**: Consulta [Métricas](docs/metricas_evaluacion.md) para interpretar resultados

---

**Nota**: Este proyecto utiliza modelos preentrenados y requiere una GPU para entrenamiento óptimo, aunque también funciona en CPU con tiempos de entrenamiento más largos.
