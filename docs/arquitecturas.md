# Arquitecturas de Modelos

Este documento explica las diferentes arquitecturas de modelos disponibles en el proyecto y sus características técnicas.

## 🏛️ Arquitecturas Disponibles

### 1. EfficientNetB3

**EfficientNet** es una familia de modelos desarrollada por Google que optimiza tanto la precisión como la eficiencia computacional.

#### Características Técnicas:
- **Input Shape**: 300x300x3 píxeles
- **Parámetros**: ~12 millones
- **Depth**: 18 capas principales
- **Width**: Factor de escalamiento 1.2
- **Resolution**: 300x300

#### Ejemplo de Implementación:
```python
base_model = EfficientNetB3(
    input_shape=(300, 300, 3),
    include_top=False,        # Sin capa de clasificación final
    weights="imagenet"        # Pesos preentrenados en ImageNet
)
base_model.trainable = False  # Congelar capas base para transfer learning
```

#### Ventajas:
- ✅ Excelente balance precisión/eficiencia
- ✅ Diseño escalable y consistente
- ✅ Estado del arte en múltiples benchmarks

#### Desventajas:
- ❌ Mayor uso de memoria que MobileNet
- ❌ Tiempo de inferencia más lento en dispositivos móviles

---

### 2. MobileNetV2

**MobileNet** está diseñado específicamente para aplicaciones móviles y dispositivos con recursos limitados.

#### Características Técnicas:
- **Input Shape**: 224x224x3 píxeles
- **Parámetros**: ~3.4 millones
- **Arquitectura**: Depthwise Separable Convolutions
- **Inverted Residuals**: Bloques de construcción principales

#### Ejemplo de Implementación:
```python
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# Augmentación de datos integrada
augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])
```

#### Ventajas:
- ✅ Muy ligero y rápido
- ✅ Ideal para dispositivos móviles
- ✅ Buen rendimiento con pocos recursos

#### Desventajas:
- ❌ Menor precisión que modelos más grandes
- ❌ Puede tener dificultades con datos complejos

---

### 3. ResNeSt50

**ResNeSt** (ResNet + Split-Attention Networks) es una mejora de ResNet que incorpora mecanismos de atención.

#### Características Técnicas:
- **Input Shape**: 224x224x3 píxeles
- **Parámetros**: ~27 millones
- **Bloques**: Split-Attention blocks
- **Profundidad**: 50 capas

#### Ejemplo de Implementación:
```python
# Usando TensorFlow Hub
resnet_url = "https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/5"
feature_extractor = hub.KerasLayer(
    resnet_url,
    input_shape=(224, 224, 3),
    trainable=False
)

model = tf.keras.Sequential([
    feature_extractor,
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax')
])
```

#### Ventajas:
- ✅ Excelente para tareas complejas
- ✅ Mecanismo de atención incorporado
- ✅ Robusta con diferentes tipos de datos

#### Desventajas:
- ❌ Mayor consumo computacional
- ❌ Más lenta en inferencia

---

### 4. ConvNeXt-Tiny

**ConvNeXt** es una arquitectura moderna que combina las fortalezas de las redes convolucionales con ideas de los Transformers.

#### Características Técnicas:
- **Input Shape**: 224x224x3 píxeles
- **Parámetros**: ~28 millones
- **Arquitectura**: Convoluciones modernizadas con técnicas de Transformers
- **Bloques**: ConvNeXt blocks con LayerScale

#### Ejemplo de Implementación:
```python
base_model = ConvNeXtTiny(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False

# Capa personalizada LayerScale
class LayerScale(layers.Layer):
    def __init__(self, init_values=1e-6, projection_dim=768, **kwargs):
        super(LayerScale, self).__init__(**kwargs)
        self.init_values = init_values
        self.projection_dim = projection_dim
```

#### Ventajas:
- ✅ Arquitectura de vanguardia (2022)
- ✅ Combina convolutiones con ideas de Transformers
- ✅ Excelente rendimiento en tareas modernas
- ✅ Diseño escalable y eficiente

#### Desventajas:
- ❌ Relativamente nuevo, menos investigado
- ❌ Requiere capas personalizadas (LayerScale)

## 🔧 Transfer Learning

Todos los modelos implementan **Transfer Learning**:

### ¿Qué es Transfer Learning?
Es una técnica donde utilizamos un modelo preentrenado (en ImageNet) y lo adaptamos para nuestra tarea específica.

### Proceso:
1. **Congelamos** las capas del modelo base (`trainable = False`)
2. **Agregamos** capas finales específicas para nuestro problema
3. **Entrenamos** solo las nuevas capas

### Ejemplo Práctico:
```python
# Modelo base congelado
base_model.trainable = False

# Nuevas capas entrenables
x = base_model(inputs, training=False)  # No entrenar durante forward pass
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)
```

## 📊 Comparación de Modelos

| Modelo | Tamaño Input | Parámetros | Velocidad | Precisión | Uso RAM |
|--------|-------------|------------|-----------|-----------|---------|
| EfficientNetB3 | 300x300 | ~12M | Medio | Alta | Alto |
| MobileNetV2 | 224x224 | ~3.4M | Rápido | Media | Bajo |
| ResNeSt50 | 224x224 | ~27M | Lento | Muy Alta | Muy Alto |
| ConvNeXt-Tiny | 224x224 | ~28M | Medio | Muy Alta | Alto |

## 🎯 Recomendaciones de Uso

### Para Producción Móvil:
```python
modelo_recomendado = "mobilenet"
```

### Para Máxima Precisión:
```python
modelo_recomendado = "resnest"  # o "convnext" para arquitecturas modernas
```

### Para Balance Óptimo:
```python
modelo_recomendado = "efficientnet"
```

### Para Investigación y Experimentación:
```python
modelo_recomendado = "convnext"  # Última generación, arquitectura híbrida
```

## 🧪 Experimentos y Ablation Studies

### Fine-tuning vs Transfer Learning:
```python
# Transfer Learning (recomendado)
base_model.trainable = False

# Fine-tuning (opcional para datos muy específicos)
base_model.trainable = True
# Usar learning rate muy bajo: 1e-5
```

### Comparación de Input Sizes:
- **224x224**: Estándar, balance velocidad/precisión
- **300x300**: Mejor para imágenes con detalles finos
- **512x512**: Máxima calidad, pero muy lento

## 📝 Consideraciones Técnicas

### Memory Management:
```python
# Configurar crecimiento de memoria GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
```

### Batch Size Optimization:
```python
# Para GPU con 8GB RAM:
batch_sizes = {
    "mobilenet": 64,      # Ligero
    "efficientnet": 32,   # Medio  
    "resnest": 16,        # Pesado
    "convnext": 32        # Medio-pesado
}
```
