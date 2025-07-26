# Preprocesamiento de Imágenes

Este documento explica las técnicas de preprocesamiento de imágenes utilizadas en el proyecto, con ejemplos prácticos y justificaciones técnicas.

## 🖼️ ¿Qué es el Preprocesamiento?

El preprocesamiento es la preparación de datos de entrada para que el modelo pueda procesarlos de manera óptima. En visión por computadora, esto incluye transformaciones como normalización, redimensionamiento y aumentación de datos.

## 🔧 Preprocesamiento en Nuestro Proyecto

### 1. Normalización de Píxeles

#### Rescaling (Método Simple)
```python
# En EfficientNet
x = layers.Rescaling(1./255)(inputs)

# Convierte valores de [0, 255] a [0, 1]
pixel_original = 128  # Valor típico de pixel
pixel_normalizado = 128 / 255 = 0.502
```

#### Preprocesamiento Específico del Modelo
```python
# MobileNet
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
x = mobilenet_preprocess(x)

# EfficientNet
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
x = efficientnet_preprocess(x)

# ConvNeXt
from tensorflow.keras.applications.convnext import preprocess_input as convnext_preprocess
x = convnext_preprocess(x)
```

### 2. Redimensionamiento Automático

```python
# Durante la carga del dataset
train_ds = image_dataset_from_directory(
    directory="data/train",
    image_size=(224, 224),  # Redimensiona automáticamente
    batch_size=32
)
```

#### ¿Por qué diferentes tamaños?

| Modelo | Tamaño | Razón |
|--------|--------|-------|
| MobileNet | 224×224 | Estándar, optimizado para velocidad |
| EfficientNet | 300×300 | Mejor para detalles finos |
| ResNeSt | 224×224 | Balance velocidad/precisión |
| ConvNeXt | 224×224 | Moderno, híbrido convolutivo-transformer |

### 3. Conversión de Formatos

```python
# En utils.py
def convert_jfif_to_jpg(path):
    """Convierte archivos .jfif a .jpg"""
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            if f.lower().endswith(".jfif"):
                src = os.path.join(dirpath, f)
                dst = os.path.join(dirpath, Path(f).stem + ".jpg")
                try:
                    with Image.open(src) as img:
                        img.convert("RGB").save(dst, "JPEG")
                    os.remove(src)
                except Exception as e:
                    print(f"Error convirtiendo {src}: {e}")
```

**Formatos soportados:**
- JPG/JPEG ✅
- PNG ✅
- JFIF → JPG (conversión automática) ✅
- BMP, TIFF, etc. (requieren conversión manual) ⚠️

## 🎨 Aumentación de Datos (Data Augmentation)

### Implementación en MobileNet

```python
# Aumentación integrada en el modelo
augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),    # Volteo horizontal
    layers.RandomRotation(0.1),         # Rotación aleatoria
    layers.RandomZoom(0.1),             # Zoom aleatorio
]) if use_augmentation else tf.keras.Sequential([])

# Aplicación en el pipeline
inputs = tf.keras.Input(shape=img_size + (3,))
x = augmentation(inputs)  # Solo durante entrenamiento
x = mobilenet_preprocess(x)
# ... resto del modelo
```

### Técnicas de Aumentación Explicadas

#### 1. Random Flip
```python
layers.RandomFlip("horizontal")

# Ejemplo visual:
# Original:    [🐶]
# Flipped:     [🶐] (espejo horizontal)
# Probabilidad: 50%
```

**¿Por qué funciona?**
- Los objetos pueden aparecer en cualquier orientación
- Duplica efectivamente el dataset
- Mejora la generalización

#### 2. Random Rotation
```python
layers.RandomRotation(0.1)  # ±10% = ±36 grados

# Ejemplo:
# factor = 0.1 significa rotación entre -36° y +36°
# factor = 0.2 sería entre -72° y +72°
```

**Casos de uso:**
- ✅ Objetos naturales (animales, plantas)
- ✅ Fotografías casuales
- ❌ Texto (se volvería ilegible)
- ❌ Objetos con orientación fija

#### 3. Random Zoom
```python
layers.RandomZoom(0.1)  # Zoom entre 90% y 110%

# height_factor y width_factor = [-0.1, 0.1]
# zoom_out: 0.9x (imagen más pequeña, padding)
# zoom_in: 1.1x (imagen más grande, recorte)
```

### Aumentación Avanzada (Opcional)

```python
# Aumentaciones adicionales que podrías agregar
advanced_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomBrightness(0.1),      # Brillo ±10%
    layers.RandomContrast(0.1),        # Contraste ±10%
    # layers.RandomTranslation(0.1, 0.1), # Traslación (TF 2.6+)
])
```

#### Ejemplo de Implementación:
```python
def create_augmented_model(base_model, img_size, num_classes, augment_strength="medium"):
    """Crea modelo con diferentes niveles de aumentación"""
    
    augmentation_configs = {
        "light": {
            "flip": True,
            "rotation": 0.05,  # ±18°
            "zoom": 0.05,      # ±5%
            "brightness": 0.0,
            "contrast": 0.0
        },
        "medium": {
            "flip": True,
            "rotation": 0.1,   # ±36°
            "zoom": 0.1,       # ±10%
            "brightness": 0.1,
            "contrast": 0.05
        },
        "heavy": {
            "flip": True,
            "rotation": 0.2,   # ±72°
            "zoom": 0.2,       # ±20%
            "brightness": 0.2,
            "contrast": 0.1
        }
    }
    
    config = augmentation_configs[augment_strength]
    
    augmentation_layers = []
    if config["flip"]:
        augmentation_layers.append(layers.RandomFlip("horizontal"))
    if config["rotation"] > 0:
        augmentation_layers.append(layers.RandomRotation(config["rotation"]))
    if config["zoom"] > 0:
        augmentation_layers.append(layers.RandomZoom(config["zoom"]))
    if config["brightness"] > 0:
        augmentation_layers.append(layers.RandomBrightness(config["brightness"]))
    if config["contrast"] > 0:
        augmentation_layers.append(layers.RandomContrast(config["contrast"]))
    
    augmentation = tf.keras.Sequential(augmentation_layers)
    
    # Construir modelo
    inputs = tf.keras.Input(shape=img_size + (3,))
    x = augmentation(inputs)
    x = base_model(x, training=False)
    # ... resto de capas
    
    return model
```

## 📊 Pipeline de Datos Optimizado

### Implementación Actual
```python
def create_datasets(temp_dir, img_size, batch_size, seed):
    """Crea datasets optimizados"""
    
    # Cargar datasets
    train_ds_raw = image_dataset_from_directory(
        f"{temp_dir}/train", 
        seed=seed, 
        image_size=img_size, 
        batch_size=batch_size
    )
    
    val_ds_raw = image_dataset_from_directory(
        f"{temp_dir}/test", 
        seed=seed, 
        image_size=img_size, 
        batch_size=batch_size
    )
    
    # Optimización de performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds_raw.prefetch(AUTOTUNE)
    val_ds = val_ds_raw.prefetch(AUTOTUNE)
    
    return train_ds, val_ds, class_names, num_classes
```

### Pipeline Optimizado Avanzado
```python
def create_optimized_pipeline(data_dir, img_size, batch_size, augment=True):
    """Pipeline de datos con optimizaciones avanzadas"""
    
    # Función de preprocesamiento
    def preprocess_image(image, label):
        # Normalizar a [0, 1]
        image = tf.cast(image, tf.float32) / 255.0
        return image, label
    
    # Función de aumentación (solo para entrenamiento)
    def augment_image(image, label):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, 0.1)
        image = tf.image.random_contrast(image, 0.9, 1.1)
        return image, label
    
    # Cargar dataset
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        f"{data_dir}/train",
        image_size=img_size,
        batch_size=batch_size,
        shuffle=True
    )
    
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        f"{data_dir}/test",
        image_size=img_size,
        batch_size=batch_size,
        shuffle=False
    )
    
    # Aplicar transformaciones
    train_ds = train_ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        train_ds = train_ds.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    val_ds = val_ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Optimizaciones de performance
    train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)
    
    return train_ds, val_ds
```

## 🔍 Análisis de Preprocesamiento

### Verificar Normalización
```python
def analyze_dataset_stats(dataset):
    """Analiza estadísticas del dataset"""
    
    pixel_values = []
    
    for images, labels in dataset.take(5):  # Tomar 5 batches
        pixel_values.extend(images.numpy().flatten())
    
    pixel_values = np.array(pixel_values)
    
    print(f"Min pixel value: {pixel_values.min():.4f}")
    print(f"Max pixel value: {pixel_values.max():.4f}")
    print(f"Mean pixel value: {pixel_values.mean():.4f}")
    print(f"Std pixel value: {pixel_values.std():.4f}")
    
    # Verificar rango esperado
    if pixel_values.min() >= 0 and pixel_values.max() <= 1:
        print("✅ Normalización correcta [0, 1]")
    elif pixel_values.min() >= -1 and pixel_values.max() <= 1:
        print("✅ Normalización correcta [-1, 1]")
    else:
        print("⚠️ Verificar normalización")

# Uso
analyze_dataset_stats(train_ds)
```

### Visualizar Aumentación
```python
def visualize_augmentation(dataset, class_names):
    """Visualiza el efecto de la aumentación"""
    
    plt.figure(figsize=(15, 10))
    
    for images, labels in dataset.take(1):
        for i in range(min(8, len(images))):
            # Imagen original (sin aumentación)
            plt.subplot(2, 4, i + 1)
            plt.imshow(images[i])
            plt.title(f"Original: {class_names[labels[i]]}")
            plt.axis('off')
            
            # Si tuviéramos aumentación aplicada, la mostraríamos aquí
    
    plt.tight_layout()
    plt.show()

# Para ver aumentación en tiempo real
def show_augmentation_effects():
    """Muestra efectos de aumentación en tiempo real"""
    
    # Cargar una imagen de ejemplo
    img_path = "data/clase1/train/ejemplo.jpg"
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) / 255.0
    
    # Crear aumentación
    augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ])
    
    # Mostrar múltiples versiones aumentadas
    plt.figure(figsize=(15, 3))
    
    # Original
    plt.subplot(1, 6, 1)
    plt.imshow(img_array[0])
    plt.title("Original")
    plt.axis('off')
    
    # 5 versiones aumentadas
    for i in range(5):
        augmented = augmentation(img_array, training=True)
        plt.subplot(1, 6, i + 2)
        plt.imshow(augmented[0])
        plt.title(f"Aumentada {i+1}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
```

## 💡 Mejores Prácticas

### 1. Cuándo Usar Cada Aumentación

```python
augmentation_guidelines = {
    "fotografías_naturales": {
        "flip": True,          # Animales, paisajes
        "rotation": 0.1,       # Objetos en cualquier orientación
        "zoom": 0.1,          # Diferentes distancias
        "brightness": 0.1      # Diferentes condiciones de luz
    },
    
    "documentos_texto": {
        "flip": False,         # Texto no se voltea
        "rotation": 0.02,      # Mínima rotación (documentos escaneados)
        "zoom": 0.05,         # Poco zoom
        "brightness": 0.05     # Mínimo cambio de brillo
    },
    
    "imagenes_medicas": {
        "flip": True,          # Anatomía puede aparecer en espejo
        "rotation": 0.05,      # Poca rotación (mantener orientación médica)
        "zoom": 0.05,         # Poco zoom (no perder detalles)
        "brightness": 0.02     # Mínimo (importante para diagnóstico)
    },
    
    "objetos_manufacturados": {
        "flip": True,          # Productos pueden aparecer en espejo
        "rotation": 0.1,       # Diferentes ángulos de vista
        "zoom": 0.15,         # Diferentes distancias de cámara
        "brightness": 0.1      # Diferentes condiciones de iluminación
    }
}
```

### 2. Evitar Aumentación Excesiva

```python
# ❌ Aumentación excesiva
bad_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomFlip("vertical"),        # Raramente útil
    layers.RandomRotation(0.5),          # ±180° demasiado
    layers.RandomZoom(0.5),              # ±50% muy extremo
    layers.RandomBrightness(0.5),        # Cambios muy drásticos
    layers.RandomContrast(0.5),
])

# ✅ Aumentación balanceada
good_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),      # 50% probabilidad
    layers.RandomRotation(0.1),          # ±36° razonable
    layers.RandomZoom(0.1),              # ±10% moderado
    layers.RandomBrightness(0.1),        # ±10% sutil
])
```

### 3. Monitoreo del Preprocesamiento

```python
def validate_preprocessing_pipeline(dataset, expected_range=(0, 1)):
    """Valida que el preprocesamiento sea correcto"""
    
    checks = {
        "range_check": False,
        "shape_check": False,
        "type_check": False,
        "nan_check": False
    }
    
    for images, labels in dataset.take(1):
        # Verificar rango de valores
        min_val, max_val = tf.reduce_min(images), tf.reduce_max(images)
        if min_val >= expected_range[0] and max_val <= expected_range[1]:
            checks["range_check"] = True
        
        # Verificar forma
        if len(images.shape) == 4:  # (batch, height, width, channels)
            checks["shape_check"] = True
        
        # Verificar tipo
        if images.dtype == tf.float32:
            checks["type_check"] = True
        
        # Verificar NaN
        if not tf.reduce_any(tf.math.is_nan(images)):
            checks["nan_check"] = True
        
        break
    
    # Imprimir resultados
    for check, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"{status} {check}: {passed}")
    
    return all(checks.values())

# Usar después de crear el dataset
is_valid = validate_preprocessing_pipeline(train_ds)
if is_valid:
    print("🎉 Pipeline de preprocesamiento válido!")
else:
    print("⚠️ Revisar pipeline de preprocesamiento")
```
