"""
Configuraciones de modelos para el entrenamiento de clasificación de imágenes.
Contiene las funciones para crear diferentes arquitecturas de modelos.
"""

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB3, MobileNetV2
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications import ConvNeXtTiny
from tensorflow.keras.applications.convnext import preprocess_input as convnext_preprocess

def create_efficientnet_model(img_size, num_classes):
    """
    Crea un modelo basado en EfficientNetB3.
    
    Args:
        img_size: Tupla con el tamaño de imagen (height, width)
        num_classes: Número de clases para clasificación
    
    Returns:
        model: Modelo de Keras compilado
        preprocess_fn: Función de preprocesamiento
        model_name: Nombre del modelo
    """
    base_model = EfficientNetB3(
        input_shape=img_size + (3,),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=img_size + (3,))
    x = layers.Rescaling(1./255)(inputs)
    x = efficientnet_preprocess(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer="adam", 
        loss="sparse_categorical_crossentropy", 
        metrics=["accuracy"]
    )

    return model, efficientnet_preprocess, "efficientnetb3"


def create_mobilenet_model(img_size, num_classes, use_augmentation=True):
    """
    Crea un modelo basado en MobileNetV2.
    
    Args:
        img_size: Tupla con el tamaño de imagen (height, width)
        num_classes: Número de clases para clasificación
        use_augmentation: Si usar aumentación de datos
    
    Returns:
        model: Modelo de Keras compilado
        preprocess_fn: Función de preprocesamiento
        model_name: Nombre del modelo
    """
    base_model = MobileNetV2(
        input_shape=img_size + (3,), 
        include_top=False, 
        weights="imagenet"
    )
    base_model.trainable = False

    # Augmentaciones
    augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ]) if use_augmentation else tf.keras.Sequential([])

    inputs = tf.keras.Input(shape=img_size + (3,))
    x = augmentation(inputs)
    x = mobilenet_preprocess(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer="adam", 
        loss="sparse_categorical_crossentropy", 
        metrics=["accuracy"]
    )

    return model, mobilenet_preprocess, "mobilenet_v2"


def create_resnest_model(img_size, num_classes):
    """
    Crea un modelo basado en ResNet desde TensorFlow Hub.
    
    Args:
        img_size: Tupla con el tamaño de imagen (height, width)
        num_classes: Número de clases para clasificación
    
    Returns:
        model: Modelo de Keras compilado
        preprocess_fn: Función de preprocesamiento (None para este modelo)
        model_name: Nombre del modelo
    """
    resnet_url = "https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/5"
    feature_extractor = hub.KerasLayer(
        resnet_url, 
        input_shape=img_size + (3,), 
        trainable=False
    )

    model = tf.keras.Sequential([
        feature_extractor,
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model, None, "resnest50"

def create_convnext_model(img_size, num_classes):
    """
    Crea un modelo basado en ConvNeXt-Tiny.
    
    Args:
        img_size: Tupla con el tamaño de imagen (height, width)
        num_classes: Número de clases para clasificación
    
    Returns:
        model: Modelo de Keras compilado
        preprocess_fn: Función de preprocesamiento
        model_name: Nombre del modelo
    """
    base_model = ConvNeXtTiny(
        input_shape=img_size + (3,),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=img_size + (3,))
    x = convnext_preprocess(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model, convnext_preprocess, "convnext_tiny"


MODEL_CONFIGS = {
    "efficientnet": {
        "img_size": (300, 300),
        "batch_size": 32,
        "epochs": 15,
        "create_fn": create_efficientnet_model,
        "description": "Modelo EfficientNetB3 con preprocesamiento de imágenes.",
    },
    "mobilenet": {
        "img_size": (224, 224),
        "batch_size": 32,
        "epochs": 15,
        "create_fn": create_mobilenet_model,
        "description": "Modelo MobileNetV2 con preprocesamiento de imágenes.",
    },
    "resnest": {
        "img_size": (224, 224),
        "batch_size": 32,
        "epochs": 15,
        "create_fn": create_resnest_model,
        "description": "Modelo ResNeSt con preprocesamiento de imágenes.",
    },
    "convnext": {
        "img_size": (224, 224),
        "batch_size": 32,
        "epochs": 15,
        "create_fn": create_convnext_model,
        "description": "Modelo ConvNeXt-Tiny con preprocesamiento de imágenes.",
    }
}

def get_available_models():
    """Retorna una lista de modelos disponibles con sus descripciones."""
    return {name: config["description"] for name, config in MODEL_CONFIGS.items()}

def get_model_config(model_name):
    """Retorna la configuración de un modelo específico."""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Modelo '{model_name}' no disponible. Modelos disponibles: {list(MODEL_CONFIGS.keys())}")
    return MODEL_CONFIGS[model_name]
