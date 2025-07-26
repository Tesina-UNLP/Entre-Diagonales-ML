"""
Utilidades comunes para el entrenamiento de modelos de clasificación de imágenes.
Contiene funciones para preparación de datos, visualización y guardado de resultados.
"""

import os
import shutil
import json
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from rich.console import Console

console = Console()

def convert_jfif_to_jpg(path):
    """Convierte archivos .jfif a .jpg en un directorio."""
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            if f.lower().endswith(".jfif"):
                src = os.path.join(dirpath, f)
                dst = os.path.join(dirpath, Path(f).stem + ".jpg")
                try:
                    with Image.open(src) as img:
                        img.convert("RGB").save(dst, "JPEG")
                    os.remove(src)
                    console.log(f"[green]Convertido:[/green] {src} → {dst}")
                except Exception as e:
                    console.log(f"[red]Error convirtiendo {src}:[/red] {e}")

def prepare_dataset_structure(data_root, temp_dir):
    """Prepara la estructura temporal del dataset para entrenamiento."""
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(f"{temp_dir}/train")
    os.makedirs(f"{temp_dir}/test")

    for class_dir in os.listdir(data_root):
        full_path = os.path.join(data_root, class_dir)
        if not os.path.isdir(full_path):
            continue
        for split in ["train", "test"]:
            src = os.path.join(full_path, split)
            dst = os.path.join(temp_dir, split, class_dir)
            if os.path.exists(src):
                shutil.copytree(src, dst)

def create_datasets(temp_dir, img_size, batch_size, seed):
    """Crea los datasets de entrenamiento y validación."""
    train_ds_raw = image_dataset_from_directory(
        f"{temp_dir}/train", seed=seed, image_size=img_size, batch_size=batch_size
    )
    val_ds_raw = image_dataset_from_directory(
        f"{temp_dir}/test", seed=seed, image_size=img_size, batch_size=batch_size
    )
    
    class_names = train_ds_raw.class_names
    num_classes = len(class_names)
    
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds_raw.prefetch(AUTOTUNE)
    val_ds = val_ds_raw.prefetch(AUTOTUNE)
    
    return train_ds, val_ds, class_names, num_classes

def plot_training_metrics(history, timestamp, plot_dir):
    """Genera y guarda las gráficas de métricas de entrenamiento."""
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    plot_subdir = os.path.join(plot_dir, timestamp)
    os.makedirs(plot_subdir, exist_ok=True)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label="Train Acc")
    plt.plot(val_acc, label="Val Acc")
    plt.legend()
    plt.title("Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(loss, label="Train Loss")
    plt.plot(val_loss, label="Val Loss")
    plt.legend()
    plt.title("Loss")

    out_path = os.path.join(plot_subdir, "training_metrics.png")
    plt.savefig(out_path)
    plt.close()
    console.log(f"[cyan]✔ Guardada curva de entrenamiento en {out_path}[/cyan]")
    return plot_subdir

def plot_confusion_matrix(model, val_ds, class_names, timestamp, plot_dir):
    """Genera y guarda la matriz de confusión."""
    y_true = []
    y_pred = []

    for images, labels in val_ds:
        preds = model.predict(images)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Matriz de Confusión")
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.tight_layout()
    
    plot_subdir = os.path.join(plot_dir, timestamp)
    os.makedirs(plot_subdir, exist_ok=True)
    
    out_path = os.path.join(plot_subdir, "confusion_matrix.png")
    plt.savefig(out_path)
    plt.close()
    console.log(f"[cyan]✔ Matriz de confusión guardada en {out_path}[/cyan]")
    
    return y_true, y_pred

def evaluate_model(model, val_ds, class_names):
    """Evalúa el modelo y retorna las predicciones y métricas."""
    y_true = []
    y_pred = []

    for images, labels in val_ds:
        preds = model.predict(images)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))

    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    f1_macro = report["macro avg"]["f1-score"]
    
    return y_true, y_pred, report, f1_macro

def save_detailed_metrics(y_true, y_pred, class_names, model_dir):
    """Guarda métricas detalladas por clase."""
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    per_class = {
        cls: {
            "precision": round(metrics["precision"], 4),
            "recall": round(metrics["recall"], 4),
            "f1": round(metrics["f1-score"], 4)
        } for cls, metrics in report.items() if cls in class_names
    }

    with open(os.path.join(model_dir, "evaluation_per_class.json"), "w") as f:
        json.dump(per_class, f, indent=2)

def save_experiment_to_csv(timestamp, history, class_names, model_dir, f1_macro):
    """Guarda el experimento en el archivo CSV histórico."""
    csv_path = Path("experiments/history.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not csv_path.exists()
    model_dir = Path(model_dir)

    final_epoch = len(history.history["accuracy"]) - 1

    row = {
        "timestamp": timestamp,
        "model_name": model_dir.name,
        "accuracy": round(history.history["accuracy"][final_epoch], 4),
        "val_accuracy": round(history.history["val_accuracy"][final_epoch], 4),
        "val_loss": round(history.history["val_loss"][final_epoch], 4),
        "f1_score": round(f1_macro, 4),
        "epochs": final_epoch + 1,
        "class_names": ",".join(class_names),
        "path": str(model_dir),
    }

    with open(csv_path, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if is_new:
            writer.writeheader()
        writer.writerow(row)

def save_model_metadata(model_dir, timestamp, history, f1_macro, class_names, base_model_name):
    """Guarda los metadatos del modelo."""
    metadata = {
        "created_at": timestamp,
        "val_accuracy": history.history["val_accuracy"][-1],
        "val_loss": history.history["val_loss"][-1],
        "f1_macro": f1_macro,
        "class_names": class_names,
        "base_model": base_model_name
    }
    
    with open(os.path.join(model_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

def save_classification_report(report, model_dir):
    """Guarda el reporte de clasificación completo."""
    with open(os.path.join(model_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)

def cleanup_temp_directory(temp_dir):
    """Limpia el directorio temporal."""
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
        console.log("[grey]Carpeta temporal eliminada.[/grey]")

def create_model_directory(model_dir_base, model_name, timestamp):
    """Crea el directorio para guardar el modelo."""
    model_dir = os.path.join(model_dir_base, f"{model_name}_{timestamp}")
    os.makedirs(model_dir, exist_ok=True)
    return model_dir

def analyze_prediction_confidence(model, test_dataset, class_names, model_dir):
    """
    Analiza la confianza de las predicciones del modelo en el dataset de prueba.
    
    Args:
        model: Modelo entrenado de Keras
        test_dataset: Dataset de prueba
        class_names: Lista con los nombres de las clases
        model_dir: Directorio donde se guardarán los resultados
        
    Returns:
        all_predictions: Lista con todas las predicciones y sus confianzas
    """
    confidences = {'high': 0, 'medium': 0, 'low': 0}
    all_predictions = []
    
    for images, labels in test_dataset:
        predictions = model.predict(images)
        
        for pred in predictions:
            max_confidence = np.max(pred)
            predicted_class = np.argmax(pred)
            
            if max_confidence > 0.9:
                confidences['high'] += 1
            elif max_confidence > 0.7:
                confidences['medium'] += 1
            else:
                confidences['low'] += 1
                
            all_predictions.append({
                'class': class_names[predicted_class],
                'confidence': max_confidence,
                'probabilities': pred
            })
    
    total = sum(confidences.values())
    print("Distribución de Confianza:")
    print(f"  Alta (>90%):   {confidences['high']}/{total} ({confidences['high']/total:.1%})")
    print(f"  Media (70-90%): {confidences['medium']}/{total} ({confidences['medium']/total:.1%})")
    print(f"  Baja (<70%):   {confidences['low']}/{total} ({confidences['low']/total:.1%})")
    
    # Guarda la confianza de las predicciones en un archivo JSON
    with open(os.path.join(model_dir, "prediction_confidence.json"), "w") as f:
        json.dump(all_predictions, f, indent=2)
    console.log("[cyan]✔ Análisis de confianza de predicciones completado.[/cyan]")
    

    return all_predictions