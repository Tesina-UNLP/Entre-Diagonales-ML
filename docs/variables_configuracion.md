# Variables y Configuraciones del Proyecto

Este documento explica todas las variables importantes, configuraciones y parámetros utilizados en el proyecto de clasificación de imágenes.

## ⚙️ Variables de Configuración Global

### Archivo: `config.py`

#### Rutas y Directorios
```python
# Directorio raíz donde se encuentran los datos organizados por clases
DATA_ROOT = "data"

# Directorio temporal para reorganizar datos durante entrenamiento
TEMP_DIR = "temp_dataset"

# Directorio base donde se guardan los modelos entrenados
MODEL_DIR_BASE = "models"

# Directorio para guardar gráficas y visualizaciones
PLOT_DIR = "plots"
```

**Explicación práctica:**
- `DATA_ROOT`: Aquí van tus carpetas de clases (ej: data/perros/, data/gatos/)
- `TEMP_DIR`: Se crea automáticamente durante entrenamiento y se borra después
- `MODEL_DIR_BASE`: Cada modelo entrenado crea una subcarpeta con timestamp
- `PLOT_DIR`: Guarda las curvas de accuracy, loss y matriz de confusión

#### Configuraciones Específicas por Modelo
```python
MODEL_SPECIFIC_CONFIG = {
    "mobilenet": {
        "img_size": (244, 244),      # Resolución de entrada
        "learning_rate": 0.001,      # Tasa de aprendizaje
        "dropout_rate": 0.2,         # Porcentaje de dropout
        "use_augementation": True,   # Usar aumentación de datos
        "dense_units": 64,           # Neuronas en capa densa
    },
    "efficientnet": {
        "img_size": (300, 300),
        "learning_rate": 0.0001,     # Más conservador
        "dropout_rate": 0.3,         # Más regularización
    },
    "resnest": {
        "img_size": (224, 224),
        "learning_rate": 0.001,
        "dropout_rate": 0.3,
        "dense_units": 250,          # Más neuronas
    },
    "convnext": {
        "img_size": (224, 224),
        "learning_rate": 0.0001,     # Conservador para modelo moderno
        "dropout_rate": 0.3,         # Regularización estándar
        "dense_units": 128,          # Balance para arquitectura híbrida
    }
}
```

## 🧠 Variables de Modelo

### Archivo: `models.py`

#### Configuración de Modelos
```python
MODEL_CONFIGS = {
    "efficientnet": {
        "img_size": (300, 300),           # Tamaño de imagen de entrada
        "batch_size": 32,                 # Imágenes por lote
        "epochs": 15,                     # Épocas de entrenamiento
        "create_fn": create_efficientnet_model,  # Función constructora
        "description": "Modelo EfficientNetB3 con preprocesamiento...",
    },
    # ... otros modelos
}
```

#### Parámetros de Arquitectura

**Input Shape (Forma de Entrada)**
```python
input_shape = img_size + (3,)  # (altura, ancho, canales RGB)

# Ejemplos:
# MobileNet: (224, 224, 3)
# EfficientNet: (300, 300, 3)
# ResNeSt: (224, 224, 3)
# ConvNeXt: (224, 224, 3)
```

**¿Por qué diferentes tamaños?**
- **224x224**: Estándar histórico, balance velocidad/calidad
- **300x300**: Mejor para detalles finos, usado en EfficientNet
- **512x512+**: Para imágenes médicas o satélites (no usado aquí)

**Dropout Rate**
```python
dropout_rate = 0.3  # 30% de neuronas se "apagan" aleatoriamente

# Efectos:
# 0.0: Sin regularización, riesgo de overfitting
# 0.2-0.3: Balance recomendado
# 0.5+: Mucha regularización, puede underfitting
```

**Learning Rate (Tasa de Aprendizaje)**
```python
learning_rates = {
    "alto": 0.01,      # Aprendizaje rápido, inestable
    "medio": 0.001,    # Balance recomendado
    "bajo": 0.0001,    # Conservador, convergencia lenta
}
```

## 🎯 Variables de Entrenamiento

### Archivo: `train.py`

#### Configuración Global de Entrenamiento
```python
CONFIG = {
    "data_root": "data",          # Directorio de datos
    "temp_dir": "temp_dataset",   # Directorio temporal
    "model_dir_base": "models",   # Modelos entrenados
    "plot_dir": "plots",          # Gráficas
    "seed": 42                    # Semilla para reproducibilidad
}
```

#### Parámetros del Dataset
```python
# En la función create_datasets()
train_ds = image_dataset_from_directory(
    directory=f"{temp_dir}/train",
    seed=42,                    # Reproducibilidad
    image_size=img_size,        # Redimensionar automáticamente
    batch_size=batch_size,      # Imágenes por lote
    validation_split=None       # Sin split adicional (ya dividido)
)
```

#### Variables de Optimización
```python
AUTOTUNE = tf.data.AUTOTUNE    # Optimización automática de pipeline

# Aplicado como:
train_ds = train_ds.prefetch(AUTOTUNE)  # Cargar datos en paralelo
```

## 📊 Variables de Evaluación

### Archivo: `utils.py`

#### Métricas de Clasificación
```python
# En evaluate_model()
report = classification_report(
    y_true, y_pred, 
    target_names=class_names, 
    output_dict=True
)

# Variables extraídas:
f1_macro = report["macro avg"]["f1-score"]  # F1 promedio
precision = report["macro avg"]["precision"] # Precisión promedio
recall = report["macro avg"]["recall"]       # Recall promedio
```

#### Variables de Visualización
```python
# Para gráficas de entrenamiento
acc = history.history["accuracy"]           # Accuracy por época
val_acc = history.history["val_accuracy"]   # Validation accuracy
loss = history.history["loss"]              # Pérdida de entrenamiento
val_loss = history.history["val_loss"]      # Pérdida de validación
```

## 🔍 Variables de Predicción

### Archivo: `predict.py`

#### Configuración de Preprocesadores
```python
PREPROCESSORS = {
    "mobilenet_v2": mobilenet_v2.preprocess_input,
    "efficientnetb3": efficientnet.preprocess_input,
    "resnet50": resnet.preprocess_input,
    "resnest50": resnet.preprocess_input,  # Mismo que ResNet
    "convnext_tiny": convnext.preprocess_input,  # Preprocesador específico ConvNeXt
}
```

#### Variables de Predicción
```python
class ModelPredictor:
    def __init__(self, model_folder):
        self.model_folder = model_folder              # Carpeta del modelo
        self.model_path = os.path.join(...)           # Ruta completa al .h5
        self.model = None                             # Modelo cargado
        self.metadata = {}                            # Metadatos del modelo
        self.class_names = []                         # Nombres de clases
        self.preprocess_fn = lambda x: x              # Función de preprocesamiento
```

## 🎨 Variables de Aumentación de Datos

### Data Augmentation en MobileNet
```python
augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),    # Volteo horizontal aleatorio
    layers.RandomRotation(0.1),         # Rotación ±10% (36°)
    layers.RandomZoom(0.1),             # Zoom ±10%
])
```

**Parámetros explicados:**
- `RandomFlip("horizontal")`: 50% probabilidad de voltear
- `RandomRotation(0.1)`: Rotar entre -36° y +36°
- `RandomZoom(0.1)`: Zoom entre 90% y 110%

## 📈 Variables de Callback y Monitoreo

### Callbacks Implícitos en el Código
```python
# Variables de historia de entrenamiento
history.history = {
    'accuracy': [0.7, 0.8, 0.85, ...],      # Lista por época
    'val_accuracy': [0.65, 0.75, 0.8, ...],
    'loss': [1.2, 0.8, 0.6, ...],
    'val_loss': [1.5, 1.0, 0.8, ...]
}
```

### Variables de Checkpoint
```python
# Timestamp para identificar experimentos
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# Ejemplo: "20240125_143052"

# Directorio del modelo
model_dir = f"{MODEL_DIR_BASE}/{model_name}_{timestamp}"
# Ejemplo: "models/efficientnet_20240125_143052"
```

## 🔧 Variables de Configuración Técnica

### GPU y Memoria
```python
# Configuración implícita de TensorFlow
tf.config.experimental.set_memory_growth(gpu, True)

# Variables de batch size por GPU
batch_sizes_recommended = {
    "GPU_4GB": {"mobilenet": 32, "efficientnet": 16, "resnest": 8, "convnext": 16},
    "GPU_8GB": {"mobilenet": 64, "efficientnet": 32, "resnest": 16, "convnext": 32},
    "GPU_16GB": {"mobilenet": 128, "efficientnet": 64, "resnest": 32},
}
```

### Variables de Archivo y Guardado
```python
# Archivos generados por modelo
files_generated = {
    "model.h5": "Modelo entrenado",
    "metadata.json": "Información del entrenamiento",
    "evaluation_per_class.json": "Métricas por clase",
    "report.json": "Reporte completo de clasificación"
}

# CSV de experimentos
experiment_columns = [
    "timestamp", "model_name", "accuracy", "val_accuracy", 
    "val_loss", "f1_score", "epochs", "class_names", "path"
]
```

## 💡 Variables de Debugging

### Variables de Monitoreo
```python
# En el entrenamiento
console.log(f"[green]✅ Modelo cargado: {model_name}[/green]")
console.log(f"[yellow]📊 Clases encontradas: {len(class_names)}[/yellow]")
console.log(f"[blue]🔄 Épocas configuradas: {epochs}[/blue]")
```

### Variables de Validación
```python
# Validación de estructura de datos
def validate_data_directory(data_root):
    classes = []
    for item in os.listdir(data_root):
        train_path = os.path.join(item_path, "train")
        test_path = os.path.join(item_path, "test")
        if os.path.exists(train_path) and os.path.exists(test_path):
            classes.append(item)
    
    return len(classes) > 0  # Boolean de validación
```

## 📋 Resumen de Variables Importantes

### Variables que PUEDES modificar:
```python
# En config.py
MODEL_SPECIFIC_CONFIG["mobilenet"]["learning_rate"] = 0.002  # Cambiar LR
MODEL_SPECIFIC_CONFIG["mobilenet"]["dropout_rate"] = 0.4     # Más regularización

# En models.py
"batch_size": 16,  # Reducir si tienes poca GPU RAM
"epochs": 25,      # Más épocas para mejor entrenamiento
```

### Variables que NO debes cambiar:
```python
# Arquitectura interna de los modelos
base_model.trainable = False  # Siempre False para transfer learning
AUTOTUNE = tf.data.AUTOTUNE   # Optimización de TensorFlow
input_shape = img_size + (3,) # Forma requerida por los modelos
```

### Variables para experimentar:
```python
# Aumentación de datos
layers.RandomRotation(0.2),     # Probar 0.0 a 0.3
layers.RandomZoom(0.15),        # Probar 0.0 a 0.2
layers.RandomBrightness(0.1),   # Agregar nueva aumentación

# Arquitectura
layers.Dense(128, activation='relu'),  # Probar 32, 64, 128, 256
layers.Dropout(0.5),                   # Probar 0.2, 0.3, 0.5
```

## 🎯 Casos de Uso Comunes

### Para Dataset Pequeño (<1000 imágenes):
```python
CONFIG_SMALL = {
    "learning_rate": 0.0001,    # Más conservador
    "dropout_rate": 0.5,        # Más regularización
    "epochs": 25,               # Más épocas
    "batch_size": 16            # Lotes más pequeños
}
```

### Para Dataset Grande (>10000 imágenes):
```python
CONFIG_LARGE = {
    "learning_rate": 0.001,     # Más agresivo
    "dropout_rate": 0.2,        # Menos regularización
    "epochs": 15,               # Menos épocas
    "batch_size": 64            # Lotes más grandes
}
```

### Para GPU Limitada:
```python
CONFIG_LOW_GPU = {
    "model": "mobilenet",       # Modelo más ligero
    "batch_size": 8,            # Lotes muy pequeños
    "img_size": (224, 224)      # Resolución estándar
}
```
