#!/usr/bin/env python3
"""
Script para probar predicciones de modelos desde la consola.
Permite cargar cualquier modelo entrenado y hacer predicciones sobre imágenes.
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow_hub as hub
from tensorflow_hub import KerasLayer
from tensorflow.keras.applications import mobilenet_v2, efficientnet, resnet
from tensorflow.keras.applications import convnext
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.utils import custom_object_scope

# Configuración de preprocesadores
PREPROCESSORS = {
    "mobilenet_v2": mobilenet_v2.preprocess_input,
    "efficientnetb3": efficientnet.preprocess_input,
    "resnet50": resnet.preprocess_input,
    "resnest50": resnet.preprocess_input,
    "convnext_tiny": convnext.preprocess_input,
}

# Definir la capa LayerScale personalizada para ConvNeXt
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
        
    def call(self, x):
        return x * self.gamma
        
    def get_config(self):
        config = super(LayerScale, self).get_config()
        config.update({
            "init_values": self.init_values,
            "projection_dim": self.projection_dim
        })
        return config

# Función para obtener objetos personalizados
def get_custom_objects():
    """Retorna diccionario con objetos personalizados para cargar modelos."""
    custom_objects = {
        'LayerScale': LayerScale,
        'KerasLayer': KerasLayer
    }
    
    # Agregar otros objetos personalizados comunes si están disponibles
    try:
        # Para modelos EfficientNet con capas personalizadas
        from tensorflow.keras.applications.efficientnet import swish
        custom_objects['swish'] = swish
    except ImportError:
        pass
    
    try:
        # Para otros modelos que puedan usar activaciones personalizadas
        import tensorflow.keras.utils as utils
        if hasattr(utils, 'get_custom_objects'):
            keras_custom = utils.get_custom_objects()
            custom_objects.update(keras_custom)
    except:
        pass
    
    return custom_objects

class ModelPredictor:
    """Clase para manejar predicciones de modelos."""
    
    def __init__(self, model_folder):
        """
        Inicializa el predictor con un modelo específico.
        
        Args:
            model_folder (str): Nombre de la carpeta del modelo en /models/
        """
        self.model_folder = model_folder

        folder_path = os.path.join("models", model_folder)
        h5_path = os.path.join(folder_path, "model.h5")

        if os.path.exists(h5_path):
            self.model_path = h5_path
        else:
            self.model_path = folder_path

        self.model = None
        self.metadata = {}
        self.class_names = []
        self.preprocess_fn = lambda x: x
        
        self._load_model_and_metadata()
    
    def _load_model_and_metadata(self):
        """Carga el modelo y su metadata asociada."""
        try:
            # Verificar que el modelo existe
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Modelo no encontrado: {self.model_path}")
            
            # Cargar el modelo con objetos personalizados
            print(f"Cargando modelo: {self.model_path}")
            
            # Intentar cargar con objetos personalizados
            try:
                with custom_object_scope(get_custom_objects()):
                    self.model = load_model(self.model_path)
            except Exception as e:
                print(f"⚠️  Error cargando con objetos personalizados: {e}")
                print("🔄 Intentando cargar sin objetos personalizados...")
                # Fallback: intentar cargar sin objetos personalizados
                self.model = load_model(self.model_path, compile=False)
            
            print(f"Modelo cargado exitosamente. Input shape: {self.model.input_shape}")
            
            # Cargar metadata
            metadata_path = os.path.join("models", self.model_folder, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, "r", encoding='utf-8') as f:
                    self.metadata = json.load(f)
                print(f"Metadata cargada: {self.metadata.get('base_model', 'Unknown')}")
            
            # Cargar nombres de clases
            self.class_names = self.metadata.get("class_names", [])
            if not self.class_names:
                # Fallback: intentar cargar desde metadata
                self.class_names = self._load_class_names_fallback()
            
            if not self.class_names:
                print("⚠️  Warning: No se encontraron nombres de clases. Usando nombres por defecto.")
                num_classes = self.model.output_shape[-1]
                self.class_names = [f"Clase_{i}" for i in range(num_classes)]
            
            print(f"Clases disponibles: {self.class_names}")
            
            # Configurar función de preprocesamiento
            base_model_name = self.metadata.get("base_model", "mobilenet_v2")
            self.preprocess_fn = PREPROCESSORS.get(base_model_name, lambda x: x)
            print(f"Preprocesador: {base_model_name}")
            
        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            sys.exit(1)
    
    def _load_class_names_fallback(self):
        """Método de respaldo para cargar nombres de clases."""
        try:
            # Intentar cargar desde el directorio de datos
            data_dir = "data"
            if os.path.exists(data_dir):
                class_names = []
                for item in os.listdir(data_dir):
                    item_path = os.path.join(data_dir, item)
                    if os.path.isdir(item_path):
                        class_names.append(item)
                return sorted(class_names)
        except Exception as e:
            print(f"Warning: Error en fallback de class_names: {e}")
        return []
    
    def predict_image(self, image_path, top_k=5):
        """
        Realiza predicción sobre una imagen.
        
        Args:
            image_path (str): Ruta a la imagen
            top_k (int): Número de predicciones top a retornar
        
        Returns:
            list: Lista de tuplas (clase, confianza)
        """
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Imagen no encontrada: {image_path}")
            
            # Obtener dimensiones esperadas del modelo
            img_height, img_width = self.model.input_shape[1:3]
            
            # Cargar y preprocesar imagen
            img = image.load_img(image_path, target_size=(img_height, img_width))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = self.preprocess_fn(img_array)
            
            # Realizar predicción
            print(f"Procesando imagen: {image_path}")
            predictions = self.model.predict(img_array, verbose=0)[0]
            
            # Obtener top_k predicciones
            top_indices = np.argsort(predictions)[-top_k:][::-1]
            
            results = []
            for i, idx in enumerate(top_indices):
                confidence = float(predictions[idx]) * 100
                class_name = self.class_names[idx] if idx < len(self.class_names) else f"Clase_{idx}"
                results.append((class_name, confidence))
                
            return results
            
        except Exception as e:
            print(f"❌ Error en predicción: {e}")
            return []
    
    def predict_batch(self, image_paths, top_k=3):
        """
        Realiza predicciones sobre múltiples imágenes.
        
        Args:
            image_paths (list): Lista de rutas de imágenes
            top_k (int): Número de predicciones top a retornar
        
        Returns:
            dict: Diccionario con resultados por imagen
        """
        results = {}
        
        for img_path in image_paths:
            print(f"\n📸 Procesando: {img_path}")
            prediction = self.predict_image(img_path, top_k)
            results[img_path] = prediction
            
            if prediction:
                print("🔍 Resultados:")
                for i, (class_name, confidence) in enumerate(prediction, 1):
                    print(f"  {i}. {class_name}: {confidence:.2f}%")
            else:
                print("❌ No se pudo procesar la imagen")
                
        return results
    
    def get_model_info(self):
        """Retorna información del modelo."""
        info = {
            "model_folder": self.model_folder,
            "model_path": self.model_path,
            "input_shape": self.model.input_shape if self.model else None,
            "output_shape": self.model.output_shape if self.model else None,
            "classes": self.class_names,
            "metadata": self.metadata
        }
        return info


def list_available_models():
    """Lista todos los modelos disponibles."""
    models_dir = "models"
    if not os.path.exists(models_dir):
        print("❌ Directorio 'models' no encontrado")
        return []
    
    available_models = []
    for folder in sorted(os.listdir(models_dir)):
        model_path = os.path.join(models_dir, folder, "model.h5")
        if os.path.exists(model_path):
            # Cargar metadata si existe
            metadata_path = os.path.join(models_dir, folder, "metadata.json")
            metadata = {}
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, "r", encoding='utf-8') as f:
                        metadata = json.load(f)
                except:
                    pass
            
            available_models.append({
                "folder": folder,
                "path": model_path,
                "base_model": metadata.get("base_model", "Unknown"),
                "accuracy": metadata.get("test_accuracy", "N/A"),
                "created": metadata.get("timestamp", "N/A")
            })
    
    return available_models


def print_model_list(models):
    """Imprime la lista de modelos disponibles."""
    if not models:
        print("❌ No se encontraron modelos disponibles")
        return
    
    print("📋 Modelos disponibles:")
    print("-" * 80)
    print(f"{'ID':<3} {'Carpeta':<25} {'Modelo Base':<15} {'Accuracy':<10} {'Fecha'}")
    print("-" * 80)
    
    for i, model in enumerate(models, 1):
        accuracy = f"{model['accuracy']:.3f}" if isinstance(model['accuracy'], (int, float)) else str(model['accuracy'])
        print(f"{i:<3} {model['folder']:<25} {model['base_model']:<15} {accuracy:<10} {model['created']}")


def main():
    """Función principal del script."""
    parser = argparse.ArgumentParser(
        description="Predictor de imágenes usando modelos entrenados",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

    # Listar modelos disponibles
    python predict_console.py --list

    # Hacer predicción con un modelo específico
    python predict_console.py --model mobilenet_v2_20250725_113354 --image test/catedral.jpg

    # Hacer predicciones sobre múltiples imágenes
    python predict_console.py --model efficientnetb3_20250725_113622 --image test/catedral.jpg test/museo.jpg

    # Hacer predicción con más resultados top
    python predict_console.py --model resnest50_20250725_113853 --image test/museo.jpg --top-k 5

    # Procesar todas las imágenes en un directorio
    python predict_console.py --model mobilenet_v2_20250725_113354 --dir test/
        """
    )
    
    parser.add_argument("--list", "-l", action="store_true", 
                       help="Lista todos los modelos disponibles")
    
    parser.add_argument("--model", "-m", type=str,
                       help="Nombre de la carpeta del modelo a usar")
    
    parser.add_argument("--image", "-i", nargs="+", 
                       help="Ruta(s) a la(s) imagen(es) a predecir")
    
    parser.add_argument("--dir", "-d", type=str,
                       help="Directorio con imágenes a procesar")
    
    parser.add_argument("--top-k", "-k", type=int, default=3,
                       help="Número de predicciones top a mostrar (default: 3)")
    
    parser.add_argument("--info", action="store_true",
                       help="Mostrar información detallada del modelo")
    
    args = parser.parse_args()
    
    # Listar modelos disponibles
    if args.list:
        models = list_available_models()
        print_model_list(models)
        return
    
    # Validar argumentos
    if not args.model:
        print("❌ Error: Debes especificar un modelo con --model")
        print("💡 Usa --list para ver modelos disponibles")
        return
    
    if not args.image and not args.dir:
        print("❌ Error: Debes especificar imagen(es) con --image o directorio con --dir")
        return
    
    # Inicializar predictor
    try:
        predictor = ModelPredictor(args.model)
    except Exception as e:
        print(f"❌ Error inicializando predictor: {e}")
        return
    
    # Mostrar información del modelo si se solicita
    if args.info:
        info = predictor.get_model_info()
        print("📊 Información del modelo:")
        print(f"  Carpeta: {info['model_folder']}")
        print(f"  Input shape: {info['input_shape']}")
        print(f"  Output shape: {info['output_shape']}")
        print(f"  Clases: {info['classes']}")
        if info['metadata']:
            print(f"  Metadata: {json.dumps(info['metadata'], indent=2, ensure_ascii=False)}")
        print()
    
    # Procesar imágenes
    image_paths = []
    
    if args.image:
        image_paths.extend(args.image)
    
    if args.dir:
        if not os.path.exists(args.dir):
            print(f"❌ Error: Directorio no encontrado: {args.dir}")
            return
        
        # Buscar imágenes en el directorio
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
        for file in os.listdir(args.dir):
            file_path = os.path.join(args.dir, file)
            if os.path.isfile(file_path) and Path(file).suffix.lower() in image_extensions:
                image_paths.append(file_path)
    
    if not image_paths:
        print("❌ Error: No se encontraron imágenes para procesar")
        return
    
    # Realizar predicciones
    print(f"🚀 Iniciando predicciones con modelo: {args.model}")
    print(f"📁 Imágenes a procesar: {len(image_paths)}")
    print("=" * 60)
    
    results = predictor.predict_batch(image_paths, args.top_k)
    
    # Resumen final
    print("\n" + "=" * 60)
    print("📈 Resumen de predicciones:")
    
    successful = sum(1 for r in results.values() if r)
    failed = len(results) - successful
    
    print(f"  ✅ Exitosas: {successful}")
    print(f"  ❌ Fallidas: {failed}")
    print(f"  📊 Total: {len(results)}")


if __name__ == "__main__":
    main()
