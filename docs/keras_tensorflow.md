# Introducción a Keras y TensorFlow

Este documento explica los conceptos fundamentales de Keras y TensorFlow utilizados en el proyecto, con ejemplos prácticos y casos de uso.

## 🔧 ¿Qué es TensorFlow?

**TensorFlow** es una plataforma de código abierto para machine learning desarrollada por Google. Proporciona un ecosistema completo de herramientas para construir y entrenar modelos de ML.

### Componentes Principales

```python
import tensorflow as tf

# Verificar versión
print(f"TensorFlow versión: {tf.__version__}")

# Verificar GPU disponible
print(f"GPU disponible: {tf.config.list_physical_devices('GPU')}")
```

## 🎭 ¿Qué es Keras?

**Keras** es una API de alto nivel integrada en TensorFlow que simplifica la construcción de redes neuronales.

### Filosofía de Keras
- **Simple**: API intuitiva y fácil de usar
- **Flexible**: Permite desde modelos simples hasta arquitecturas complejas
- **Potente**: Escalable desde investigación hasta producción

## 🏗️ Conceptos Fundamentales

### 1. Tensores

Los **tensores** son arrays multidimensionales, la estructura de datos básica en TensorFlow.

```python
import tensorflow as tf
import numpy as np

# Escalar (tensor 0D)
escalar = tf.constant(42)
print(f"Escalar: {escalar}")

# Vector (tensor 1D)
vector = tf.constant([1, 2, 3, 4])
print(f"Vector: {vector}")

# Matriz (tensor 2D)
matriz = tf.constant([[1, 2], [3, 4]])
print(f"Matriz:\n{matriz}")

# Imagen (tensor 3D: altura, ancho, canales)
imagen = tf.random.normal([224, 224, 3])
print(f"Forma de imagen: {imagen.shape}")

# Batch de imágenes (tensor 4D: batch, altura, ancho, canales)
batch_imagenes = tf.random.normal([32, 224, 224, 3])
print(f"Forma del batch: {batch_imagenes.shape}")
```

### 2. Capas (Layers)

Las **capas** son los bloques de construcción básicos de las redes neuronales.

```python
from tensorflow.keras import layers

# Capa densa (completamente conectada)
dense_layer = layers.Dense(
    units=64,           # Número de neuronas
    activation='relu',  # Función de activación
    input_shape=(784,)  # Forma de entrada (solo para primera capa)
)

# Capa convolucional
conv_layer = layers.Conv2D(
    filters=32,         # Número de filtros
    kernel_size=3,      # Tamaño del kernel
    activation='relu',
    input_shape=(28, 28, 1)
)

# Capa de pooling
pool_layer = layers.MaxPooling2D(pool_size=2)

# Capa de dropout (regularización)
dropout_layer = layers.Dropout(rate=0.3)
```

### 3. Modelos

Los **modelos** organizan las capas en una arquitectura específica.

#### Modelo Secuencial (Más Simple)
```python
from tensorflow.keras.models import Sequential

# Modelo para clasificación de imágenes MNIST
model_simple = Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

model_simple.summary()
```

#### Functional API (Más Flexible)
```python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

# Entrada
inputs = Input(shape=(224, 224, 3))

# Capas
x = layers.Conv2D(32, 3, activation='relu')(inputs)
x = layers.MaxPooling2D(2)(x)
x = layers.Conv2D(64, 3, activation='relu')(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(64, activation='relu')(x)
outputs = layers.Dense(10, activation='softmax')(x)

# Modelo
model_funcional = Model(inputs=inputs, outputs=outputs)
```

## 🎯 Casos de Uso en Nuestro Proyecto

### 1. Transfer Learning con Modelos Preentrenados

```python
from tensorflow.keras.applications import EfficientNetB3

# Cargar modelo preentrenado
base_model = EfficientNetB3(
    input_shape=(300, 300, 3),
    include_top=False,      # Sin la capa de clasificación
    weights='imagenet'      # Pesos de ImageNet
)

# Congelar capas base
base_model.trainable = False

# Construir modelo completo
inputs = tf.keras.Input(shape=(300, 300, 3))
x = layers.Rescaling(1./255)(inputs)                    # Normalización
x = base_model(x, training=False)                       # Modelo base
x = layers.GlobalAveragePooling2D()(x)                  # Reducir dimensiones
x = layers.Dropout(0.3)(x)                              # Regularización
outputs = layers.Dense(num_classes, activation='softmax')(x)  # Clasificación

model = Model(inputs, outputs)
```

### 2. Compilación del Modelo

```python
model.compile(
    optimizer='adam',                           # Algoritmo de optimización
    loss='sparse_categorical_crossentropy',    # Función de pérdida
    metrics=['accuracy']                        # Métricas a monitorear
)
```

#### Explicación de Parámetros:

**Optimizer (adam)**:
- Algoritmo de optimización adaptativo
- Combina momentum con learning rate adaptativo
- Buen rendimiento general

**Loss (sparse_categorical_crossentropy)**:
- Para clasificación multiclase
- Labels como enteros (0, 1, 2, ...) no one-hot
- Calcula la diferencia entre predicciones y etiquetas reales

**Metrics (accuracy)**:
- Porcentaje de predicciones correctas
- Fácil de interpretar

### 3. Entrenamiento

```python
# Entrenar el modelo
history = model.fit(
    train_dataset,           # Datos de entrenamiento
    validation_data=val_dataset,  # Datos de validación
    epochs=15,               # Número de épocas
    verbose=1,               # Mostrar progreso
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=3),
        tf.keras.callbacks.ReduceLROnPlateau(patience=2)
    ]
)
```

## 📊 Manipulación de Datos

### 1. Datasets con tf.data

```python
# Crear dataset desde directorio
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'data/train',
    image_size=(224, 224),
    batch_size=32,
    seed=42
)

# Optimizar performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)

# Aplicar transformaciones
def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0  # Normalizar
    return image, label

train_ds = train_ds.map(preprocess)
```

### 2. Aumentación de Datos

```python
# Aumentación integrada en el modelo
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# Aplicar en el modelo
inputs = tf.keras.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)
# ... resto del modelo
```

## 🔍 Debugging y Monitoreo

### 1. Visualizar Arquitectura

```python
# Resumen del modelo
model.summary()

# Visualizar gráficamente
tf.keras.utils.plot_model(
    model, 
    to_file='model.png', 
    show_shapes=True,
    show_layer_names=True
)
```

### 2. Callbacks Útiles

```python
callbacks = [
    # Guardar mejor modelo
    tf.keras.callbacks.ModelCheckpoint(
        'best_model.h5',
        save_best_only=True,
        monitor='val_accuracy'
    ),
    
    # Parada temprana
    tf.keras.callbacks.EarlyStopping(
        patience=5,
        monitor='val_loss',
        restore_best_weights=True
    ),
    
    # Reducir learning rate
    tf.keras.callbacks.ReduceLROnPlateau(
        patience=3,
        factor=0.5,
        min_lr=1e-7
    ),
    
    # Logging personalizado
    tf.keras.callbacks.CSVLogger('training.log')
]
```

### 3. Monitorear Entrenamiento

```python
import matplotlib.pyplot as plt

def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy
    ax1.plot(history.history['accuracy'], label='Training')
    ax1.plot(history.history['val_accuracy'], label='Validation')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # Loss
    ax2.plot(history.history['loss'], label='Training')
    ax2.plot(history.history['val_loss'], label='Validation')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

# Usar después del entrenamiento
plot_training_history(history)
```

## 🚀 Predicciones

### 1. Predicción Simple

```python
# Cargar modelo entrenado
model = tf.keras.models.load_model('mi_modelo.h5')

# Cargar y procesar imagen
img = tf.keras.preprocessing.image.load_img(
    'imagen.jpg', 
    target_size=(224, 224)
)
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Crear batch

# Predicción
predictions = model.predict(img_array)
predicted_class = tf.argmax(predictions[0])

print(f"Clase predicha: {predicted_class}")
print(f"Confianza: {tf.reduce_max(predictions[0]):.2%}")
```

### 2. Predicción por Lotes

```python
# Procesar múltiples imágenes
def predict_batch(model, image_paths, class_names):
    batch_images = []
    
    for path in image_paths:
        img = tf.keras.preprocessing.image.load_img(
            path, target_size=(224, 224)
        )
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = img_array / 255.0  # Normalizar
        batch_images.append(img_array)
    
    batch_images = tf.stack(batch_images)
    predictions = model.predict(batch_images)
    
    results = []
    for i, pred in enumerate(predictions):
        class_idx = tf.argmax(pred)
        confidence = tf.reduce_max(pred)
        
        results.append({
            'path': image_paths[i],
            'class': class_names[class_idx],
            'confidence': float(confidence)
        })
    
    return results
```

## 💡 Mejores Prácticas

### 1. Gestión de Memoria

```python
# Configurar crecimiento de memoria GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
```

### 2. Reproducibilidad

```python
import numpy as np
import random

# Fijar semillas para reproducibilidad
def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seeds(42)
```

### 3. Validación del Modelo

```python
# Evaluación completa
def evaluate_model(model, test_dataset, class_names):
    # Métricas básicas
    loss, accuracy = model.evaluate(test_dataset)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Predicciones detalladas
    y_true = []
    y_pred = []
    
    for images, labels in test_dataset:
        predictions = model.predict(images)
        y_true.extend(labels.numpy())
        y_pred.extend(tf.argmax(predictions, axis=1).numpy())
    
    # Reporte de clasificación
    from sklearn.metrics import classification_report
    print("\nReporte de Clasificación:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    return y_true, y_pred
```

## 🎓 Conceptos Avanzados

### 1. Capas Personalizadas

```python
class CustomLayer(tf.keras.layers.Layer):
    def __init__(self, units=32):
        super(CustomLayer, self).__init__()
        self.units = units
    
    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='random_normal',
            trainable=True
        )
        self.b = self.add_weight(
            shape=(self.units,),
            initializer='zero',
            trainable=True
        )
    
    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b
```

### 2. Métricas Personalizadas

```python
class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)
    
    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return 2 * ((p * r) / (p + r + tf.keras.backend.epsilon()))
    
    def reset_states(self):
        self.precision.reset_states()
        self.recall.reset_states()
```
