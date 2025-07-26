"""
Configuración específica para la aplicación Flask.
"""

import os
from pathlib import Path

# Configuración básica de Flask
class FlaskConfig:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    UPLOAD_FOLDER = 'uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    
    # Configuración de modelos
    MODEL_DIR = 'models'
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
    
    # Configuración de la aplicación
    DEBUG = os.environ.get('FLASK_ENV') == 'development'
    HOST = os.environ.get('FLASK_HOST') or '0.0.0.0'
    PORT = int(os.environ.get('FLASK_PORT') or 5000)

# Mapeo de clases para los edificios de La Plata
# Actualizar según tu dataset específico
CLASS_NAMES = {
    0: 'Casa Curutchet',
    1: 'Catedral',
    2: 'Dardo Rocha',
    3: 'Municipalidad',
    4: 'Museo de Ciencias Naturales',
    5: 'Teatro Argentino'
}

# Configuración específica por tipo de modelo
MODEL_CONFIGS = {
    'mobilenet_v2': {
        'img_size': (224, 224),
        'preprocess': 'mobilenet_v2',
        'description': 'MobileNetV2 - Ligero y eficiente'
    },
    'efficientnetb3': {
        'img_size': (300, 300),
        'preprocess': 'efficientnet',
        'description': 'EfficientNetB3 - Balance entre precisión y velocidad'
    },
    'convnext_tiny': {
        'img_size': (224, 224),
        'preprocess': 'convnext',
        'description': 'ConvNeXt Tiny - Arquitectura moderna'
    },
    'resnet50': {
        'img_size': (224, 224),
        'preprocess': 'resnet',
        'description': 'ResNet50 - Clásico y robusto'
    }
}

# Mensajes de error personalizados
ERROR_MESSAGES = {
    'no_file': 'No se seleccionó ningún archivo',
    'invalid_file': 'Tipo de archivo no válido. Use: ' + ', '.join(FlaskConfig.ALLOWED_EXTENSIONS),
    'file_too_large': f'Archivo demasiado grande. Máximo: {FlaskConfig.MAX_CONTENT_LENGTH // (1024*1024)}MB',
    'no_model': 'No hay modelo cargado. Entrene un modelo primero.',
    'prediction_error': 'Error al procesar la imagen',
    'model_load_error': 'Error al cargar el modelo'
}

# Configuración de logging
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'default': {
            'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
        }
    },
    'handlers': {
        'wsgi': {
            'class': 'logging.StreamHandler',
            'stream': 'ext://flask.logging.wsgi_errors_stream',
            'formatter': 'default'
        }
    },
    'root': {
        'level': 'INFO',
        'handlers': ['wsgi']
    }
}

def ensure_directories():
    """Crea los directorios necesarios si no existen."""
    directories = [
        FlaskConfig.UPLOAD_FOLDER,
        FlaskConfig.MODEL_DIR,
        'logs'  # Para logs futuros
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)

def get_model_info(model_path):
    """Extrae información del modelo basada en el nombre del archivo."""
    model_name = Path(model_path).stem.lower()
    
    for key, config in MODEL_CONFIGS.items():
        if key.replace('_', '') in model_name.replace('_', ''):
            return {
                'type': key,
                'config': config,
                'file': model_path
            }
    
    # Default configuration
    return {
        'type': 'unknown',
        'config': MODEL_CONFIGS['efficientnetb3'],  # Default
        'file': model_path
    }
