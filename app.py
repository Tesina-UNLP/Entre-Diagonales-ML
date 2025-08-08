#!/usr/bin/env python3
"""
Aplicación Flask para predicción de imágenes arquitectónicas.
Permite subir imágenes y obtener las predicciones del top 5 de clases.
"""

import os
import json
import numpy as np
from flask import Flask, request, render_template, jsonify, flash, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import mobilenet_v2, efficientnet, convnext
from tensorflow.keras.utils import custom_object_scope
from tensorflow.keras import layers
import tensorflow_hub as hub
from pathlib import Path

# Configuración de la aplicación
app = Flask(__name__)
app.config['SECRET_KEY'] = 'tu_clave_secreta_aqui'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Extensiones permitidas
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Mapeo de clases (edificios de La Plata)
CLASS_NAMES = {
    0: 'Casa Curutchet',
    1: 'Catedral',
    2: 'Dardo Rocha',
    3: 'Municipalidad',
    4: 'Museo de Ciencias Naturales',
    5: 'Teatro Argentino'
}

# Configuración de preprocesadores
PREPROCESSORS = {
    "mobilenet_v2": mobilenet_v2.preprocess_input,
    "efficientnetb3": efficientnet.preprocess_input,
    "convnext_tiny": convnext.preprocess_input,
}

class LayerScale(layers.Layer):
    """Capa LayerScale personalizada para ConvNeXt."""
    
    def __init__(self, init_values=1e-6, projection_dim=768, **kwargs):
        super(LayerScale, self).__init__(**kwargs)
        self.init_values = init_values
        self.projection_dim = projection_dim
        
    def build(self, input_shape):
        self.gamma = self.add_weight(
            shape=(self.projection_dim,),
            initializer="zeros",
            trainable=True,
            name="gamma"
        )
        self.gamma.assign(tf.ones_like(self.gamma) * self.init_values)
        super(LayerScale, self).build(input_shape)
        
    def call(self, inputs):
        return inputs * self.gamma
        
    def get_config(self):
        config = super(LayerScale, self).get_config()
        config.update({
            "init_values": self.init_values,
            "projection_dim": self.projection_dim
        })
        return config

# Variable global para el modelo
model = None
model_name = None
img_size = None

def allowed_file(filename):
    """Verifica si el archivo tiene una extensión permitida."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_available_models():
    """Obtiene lista de modelos disponibles."""
    models_dir = Path("models")
    if not models_dir.exists():
        return []
    
    available_models = []
    for model_dir in models_dir.iterdir():
        if model_dir.is_dir():
            model_file = model_dir / "model.h5"
            if model_file.exists():
                # Leer metadata si existe
                metadata_file = model_dir / "metadata.json"
                metadata = {}
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                    except:
                        pass
                
                available_models.append({
                    'path': str(model_file.resolve()),
                    'name': model_dir.name,
                    'metadata': metadata,
                    'created': model_file.stat().st_ctime
                })
    
    # Ordenar por fecha de creación (más reciente primero)
    available_models.sort(key=lambda x: x['created'], reverse=True)
    return available_models

def load_trained_model(model_path=None):
    """Carga el modelo entrenado especificado o el más reciente."""
    global model, model_name, img_size
    
    if model_path is None:
        # Cargar el modelo más reciente
        available_models = get_available_models()
        if not available_models:
            return False, "No se encontraron modelos entrenados"
        model_path = available_models[0]['path']
    
    try:
        # Determinar el tipo de modelo por el nombre del archivo
        model_filename = Path(model_path).parent.name.lower()
        
        if "mobilenet" in model_filename:
            model_name = "mobilenet_v2"
            img_size = (244, 244)
        elif "efficientnet" in model_filename:
            model_name = "efficientnetb3"
            img_size = (300, 300)
        elif "convnext" in model_filename:
            model_name = "convnext_tiny"
            img_size = (224, 224)
        else:
            model_name = "efficientnetb3"  # Por defecto
            img_size = (300, 300)
        
        # Cargar modelo con objetos personalizados si es necesario
        custom_objects = {
            'KerasLayer': hub.KerasLayer,
            'LayerScale': LayerScale
        }
        
        # Normalizar la ruta del modelo
        model_path_normalized = str(Path(model_path).resolve())
        
        with custom_object_scope(custom_objects):
            model = load_model(model_path_normalized)
        
        return True, f"Modelo cargado: {Path(model_path).parent.name}"
        
    except Exception as e:
        return False, f"Error al cargar el modelo: {str(e)}"

def preprocess_image(img_path, target_size):
    """Preprocesa la imagen para predicción."""
    try:
        # Cargar y redimensionar imagen
        img = image.load_img(img_path, target_size=target_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Aplicar preprocesamiento específico del modelo
        if model_name in PREPROCESSORS:
            img_array = PREPROCESSORS[model_name](img_array)
        else:
            img_array = img_array / 255.0  # Normalización básica
        
        return img_array
    except Exception as e:
        raise Exception(f"Error al procesar imagen: {str(e)}")

def predict_image(img_path):
    """Realiza predicción sobre una imagen."""
    if model is None:
        return None, "Modelo no cargado"
    
    try:
        # Preprocesar imagen
        processed_img = preprocess_image(img_path, img_size)
        
        # Realizar predicción
        predictions = model.predict(processed_img)
        
        # Obtener top 5 predicciones
        top_indices = np.argsort(predictions[0])[::-1][:5]
        top_predictions = []
        
        for i, idx in enumerate(top_indices):
            class_name = CLASS_NAMES.get(idx, f"Clase {idx}")
            confidence = float(predictions[0][idx]) * 100
            top_predictions.append({
                'rank': i + 1,
                'class': class_name,
                'confidence': confidence
            })
        
        return top_predictions, None
        
    except Exception as e:
        return None, f"Error en predicción: {str(e)}"

@app.route('/')
def index():
    """Página principal."""
    available_models = get_available_models()
    return render_template('index.html', 
                         class_names=CLASS_NAMES,
                         models=available_models,
                         current_model=model_name if model else None)

@app.route('/load_model', methods=['POST'])
def load_model_route():
    """Carga un modelo específico."""
    model_path = request.form.get('model_path')
    if not model_path:
        flash('No se especificó un modelo')
        return redirect(url_for('index'))
    
    success, message = load_trained_model(model_path)
    if success:
        flash(f'✓ {message}', 'success')
    else:
        flash(f'✗ {message}', 'error')
    
    return redirect(url_for('index'))

@app.route('/upload', methods=['POST'])
def upload_file():
    """Maneja la subida de archivos y predicción."""
    if 'file' not in request.files:
        flash('No se seleccionó ningún archivo')
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        flash('No se seleccionó ningún archivo')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        
        # Crear directorio de uploads si no existe
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Realizar predicción
        predictions, error = predict_image(filepath)
        
        if error:
            flash(f'Error en predicción: {error}')
            return redirect(url_for('index'))
        
        # Limpiar archivo temporal
        try:
            os.remove(filepath)
        except:
            pass
        
        return render_template('results.html', 
                             predictions=predictions, 
                             filename=filename,
                             model_name=model_name)
    
    flash('Tipo de archivo no permitido')
    return redirect(url_for('index'))

@app.route('/upload', methods=['GET'])
def upload_get():
    """Redirige a la página principal si se accede por GET a /upload."""
    flash('Usa el formulario de la página inicial para subir una imagen.', 'error')
    return redirect(url_for('index'))

@app.errorhandler(413)
def too_large(e):
    """Manejo de archivos demasiado grandes."""
    flash('La imagen supera el tamaño máximo permitido (16MB).', 'error')
    return redirect(url_for('index'))

@app.errorhandler(405)
def method_not_allowed(e):
    """Manejo de métodos no permitidos, redirigiendo al inicio."""
    return redirect(url_for('index'))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint para predicción."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file'}), 400
    
    filename = secure_filename(file.filename)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    predictions, error = predict_image(filepath)
    
    # Limpiar archivo temporal
    try:
        os.remove(filepath)
    except:
        pass
    
    if error:
        return jsonify({'error': error}), 500
    
    return jsonify({
        'predictions': predictions,
        'model_name': model_name,
        'success': True
    })

@app.route('/health')
def health():
    """Endpoint de salud."""
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'model_name': model_name,
        'available_models': len(get_available_models())
    })

@app.route('/api/models')
def api_models():
    """API endpoint para obtener modelos disponibles."""
    return jsonify({
        'models': get_available_models(),
        'current_model': model_name if model else None
    })

if __name__ == '__main__':
    # Crear directorio de uploads
    os.makedirs('uploads', exist_ok=True)
    
    # Intentar cargar modelo
    success, message = load_trained_model()
    if success:
        print(f"✓ {message}")
    else:
        print(f"⚠ {message}")
        print("La aplicación funcionará pero necesitarás entrenar un modelo primero.")
    
    # Ejecutar aplicación
    print("\n🚀 Iniciando aplicación Flask...")
    print("📱 Accede a: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
