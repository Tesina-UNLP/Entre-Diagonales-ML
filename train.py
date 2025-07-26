#!/usr/bin/env python3
"""
Script principal para entrenar modelos de clasificación de imágenes.
Permite seleccionar entre diferentes arquitecturas de modelos.

Uso:
    python train.py --model efficientnet
    python train.py --model mobilenet
    python train.py --model resnest
    python train.py --list-models
"""

import argparse
import os
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.prompt import Confirm
import tensorflow as tf

# Importar módulos locales
from utils import (
    convert_jfif_to_jpg,
    prepare_dataset_structure,
    create_datasets,
    plot_training_metrics,
    plot_confusion_matrix,
    evaluate_model,
    save_detailed_metrics,
    save_experiment_to_csv,
    save_model_metadata,
    save_classification_report,
    cleanup_temp_directory,
    create_model_directory,
    analyze_prediction_confidence
)
from models import get_available_models, get_model_config

console = Console()

# Configuración global
CONFIG = {
    "data_root": "data",
    "temp_dir": "temp_dataset",
    "model_dir_base": "models",
    "plot_dir": "plots",
    "seed": 42
}

def show_available_models():
    """Muestra los modelos disponibles en una tabla."""
    models = get_available_models()
    
    table = Table(title="Modelos Disponibles")
    table.add_column("Modelo", style="cyan", no_wrap=True)
    table.add_column("Descripción", style="white")
    
    for name, description in models.items():
        table.add_row(name, description)
    
    console.print(table)

def validate_data_directory(data_root):
    """Valida que el directorio de datos existe y tiene la estructura correcta."""
    if not os.path.exists(data_root):
        console.print(f"[red]Error:[/red] El directorio '{data_root}' no existe.")
        return False
    
    classes = []
    for item in os.listdir(data_root):
        item_path = os.path.join(data_root, item)
        if os.path.isdir(item_path):
            train_path = os.path.join(item_path, "train")
            test_path = os.path.join(item_path, "test")
            if os.path.exists(train_path) and os.path.exists(test_path):
                classes.append(item)
    
    if not classes:
        console.print(f"[red]Error:[/red] No se encontraron clases válidas en '{data_root}'.")
        console.print("Estructura esperada: data/clase/train/ y data/clase/test/")
        return False
    
    console.print(f"[green]✓[/green] Directorio de datos válido con {len(classes)} clases: {', '.join(classes)}")
    return True

def train_model(model_name, custom_epochs=None):
    """
    Entrena un modelo específico.
    
    Args:
        model_name: Nombre del modelo a entrenar
        custom_epochs: Número de épocas personalizado (opcional)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Obtener configuración del modelo
    try:
        model_config = get_model_config(model_name)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        return False
    
    # Usar epochs personalizado si se proporciona
    epochs = custom_epochs if custom_epochs is not None else model_config["epochs"]
    
    console.rule(f"[bold green]Entrenando modelo: {model_name.upper()}")
    console.print(f"[blue]Configuración:[/blue]")
    console.print(f"  • Tamaño de imagen: {model_config['img_size']}")
    console.print(f"  • Batch size: {model_config['batch_size']}")
    console.print(f"  • Épocas: {epochs}")
    console.print(f"  • Descripción: {model_config['description']}")
    console.print()
    
    # Validar directorio de datos
    if not validate_data_directory(CONFIG["data_root"]):
        return False
    
    # Confirmar entrenamiento
    if not Confirm.ask("¿Continuar con el entrenamiento?"):
        console.print("[yellow]Entrenamiento cancelado.[/yellow]")
        return False
    
    try:
        # Preparar datos
        console.log("[yellow]📁 Preparando datos...[/yellow]")
        convert_jfif_to_jpg(CONFIG["data_root"])
        prepare_dataset_structure(CONFIG["data_root"], CONFIG["temp_dir"])
        
        # Crear datasets
        train_ds, val_ds, class_names, num_classes = create_datasets(
            CONFIG["temp_dir"],
            model_config["img_size"],
            model_config["batch_size"],
            CONFIG["seed"]
        )
        
        console.log(f"[green]✓[/green] Datasets creados: {num_classes} clases ({', '.join(class_names)})")
        
        # Crear modelo
        console.log("[yellow]🏗️ Creando modelo...[/yellow]")
        model, preprocess_fn, model_base_name = model_config["create_fn"](
            model_config["img_size"], 
            num_classes
        )
        
        # Crear directorio para guardar el modelo
        model_dir = create_model_directory(CONFIG["model_dir_base"], model_base_name, timestamp)
        model_out = os.path.join(model_dir, "model.h5")
        
        console.log(f"[green]✓[/green] Modelo creado: {model_base_name}")
        console.log(f"[cyan]📁 Directorio de salida: {model_dir}[/cyan]")
        
        # Entrenar modelo
        console.log("[yellow]🚀 Iniciando entrenamiento...[/yellow]")
                
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            verbose=1,
        )
        console.log("[green]✅ Entrenamiento completado.[/green]")
        
        # Guardar modelo
        model.save(model_dir)
        console.log(f"[bold green]✅ Modelo guardado en {model_dir}[/bold green]")
                
        # Generar visualizaciones
        console.log("[yellow]📊 Generando visualizaciones...[/yellow]")
        plot_training_metrics(history, timestamp, CONFIG["plot_dir"])
        y_true, y_pred = plot_confusion_matrix(model, val_ds, class_names, timestamp, CONFIG["plot_dir"])
        
        # Evaluar modelo
        console.log("[yellow]📈 Evaluando modelo...[/yellow]")
        y_true_eval, y_pred_eval, report, f1_macro = evaluate_model(model, val_ds, class_names)
        
        # Guardar resultados
        save_detailed_metrics(y_true_eval, y_pred_eval, class_names, model_dir)
        save_experiment_to_csv(timestamp, history, class_names, model_dir, f1_macro)
        save_model_metadata(model_dir, timestamp, history, f1_macro, class_names, model_base_name)
        save_classification_report(report, model_dir)
        
        # Mostrar resumen final
        console.rule("[bold green]Resumen del Entrenamiento")
        final_acc = history.history["accuracy"][-1]
        final_val_acc = history.history["val_accuracy"][-1]
        final_val_loss = history.history["val_loss"][-1]
        
        console.print(f"[green]✅ Entrenamiento completado exitosamente[/green]")
        console.print(f"  • Accuracy final: {final_acc:.4f}")
        console.print(f"  • Validation accuracy: {final_val_acc:.4f}")
        console.print(f"  • Validation loss: {final_val_loss:.4f}")
        console.print(f"  • F1-score macro: {f1_macro:.4f}")
        console.print(f"  • Modelo guardado en: {model_dir}")
        console.print(f"  • Gráficas guardadas en: plots/{timestamp}")

        analyze_prediction_confidence(model, val_ds, class_names, model_dir)
        
        model.summary()

        tf.keras.utils.plot_model(
            model, 
            to_file=os.path.join(model_dir, "model_architecture.png"),
            show_shapes=True,
            show_layer_names=True
        )

        return True
        
    except Exception as e:
        console.print(f"[red]Error durante el entrenamiento:[/red] {e}")
        return False
    
    finally:
        # Limpiar directorio temporal
        cleanup_temp_directory(CONFIG["temp_dir"])

def main():
    """Función principal del script."""
    parser = argparse.ArgumentParser(
        description="Entrena modelos de clasificación de imágenes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Ejemplos de uso:
    python train.py --model efficientnet          # Entrenar EfficientNetB3
    python train.py --model mobilenet --epochs 20 # Entrenar MobileNet con 20 épocas
    python train.py --list-models                 # Mostrar modelos disponibles
            """
        )
    
    parser.add_argument(
        "--model",
        type=str,
        help="Modelo a entrenar (efficientnet, mobilenet, resnest)"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        help="Número de épocas de entrenamiento (sobrescribe la configuración por defecto)"
    )
    
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Muestra los modelos disponibles"
    )
    
    args = parser.parse_args()
    
    # Mostrar header
    console.rule("[bold blue]Entrenador de Modelos de Clasificación")
    
    if args.list_models:
        show_available_models()
        return
    
    if not args.model:
        console.print("[red]Error:[/red] Debe especificar un modelo con --model")
        console.print("Use --list-models para ver los modelos disponibles")
        return
    
    # Entrenar modelo
    success = train_model(args.model, args.epochs)
    
    if success:
        console.print("\n[bold green]🎉 ¡Entrenamiento finalizado con éxito![/bold green]")
    else:
        console.print("\n[bold red]❌ El entrenamiento falló.[/bold red]")

if __name__ == "__main__":
    main()
