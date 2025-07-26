import os
from pathlib import Path

# Dataset
DATA_ROOT = "data"
TEMP_DIR = "temp_dataset"

# Donde apareceran los modelos entrenados
MODEL_DIR_BASE = "models"

# Donde se guardaran los graficos de los modelos
PLOT_DIR = "plots"

MODEL_SPECIFIC_CONFIG = {
    "mobilenet": {
        "img_size": (244, 244),
        "learning_rate": 0.001,
        "dropout_rate": 0.2,
        "use_augementation": True,
        "dense_units": 64,
    },
    "efficientnet": {
        "img_size": (300, 300),
        "learning_rate": 0.0001,
        "dropout_rate": 0.3,
    },
    "resnest": {
        "img_size": (224, 224),
        "learning_rate": 0.001,
        "dropout_rate": 0.3,
        "dense_units": 250,
    },
    "convnext": {
        "img_size": (224, 224),
        "learning_rate": 0.0001,
        "dropout_rate": 0.3,
        "dense_units": 128,
    }
}

def ensure_directories():
    """Crea los directorios necesarios si no existen."""
    directories = [MODEL_DIR_BASE, PLOT_DIR]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
