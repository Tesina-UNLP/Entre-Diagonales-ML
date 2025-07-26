"""
Script para analizar la distribución del dataset de arquitectura
Este script revisa la estructura de directorios y archivos en el dataset,
considerando extensiones de archivos case-sensitive.
"""

import os
import glob
from collections import defaultdict, Counter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def get_valid_image_extensions():
    """Retorna una lista de extensiones de imagen válidas (case-sensitive)"""
    return ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.JPG', '.JPEG', '.PNG', '.BMP', '.TIFF', '.TIF']

def format_size(bytes_size):
    """Convierte bytes a formato legible (KB, MB, GB)"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f} TB"

def analyze_dataset_structure(data_dir):
    """
    Analiza la estructura del dataset y retorna estadísticas detalladas
    
    Args:
        data_dir (str): Ruta al directorio de datos
        
    Returns:
        dict: Diccionario con estadísticas del dataset
    """
    results = {
        'classes': [],
        'train_counts': {},
        'test_counts': {},
        'total_counts': {},
        'train_sizes': {},
        'test_sizes': {},
        'total_sizes': {},
        'file_extensions': Counter(),
        'invalid_files': [],
        'directory_structure': {}
    }
    
    valid_extensions = get_valid_image_extensions()
    
    if not os.path.exists(data_dir):
        print(f"Error: El directorio {data_dir} no existe")
        return results
    
    # Obtener todas las clases (subdirectorios)
    classes = [d for d in os.listdir(data_dir) 
               if os.path.isdir(os.path.join(data_dir, d))]
    classes.sort()
    results['classes'] = classes
    
    print(f"Clases encontradas: {len(classes)}")
    print(f"Clases: {', '.join(classes)}")
    print("-" * 60)
    
    for class_name in classes:
        class_dir = os.path.join(data_dir, class_name)
        results['directory_structure'][class_name] = {}
        
        # Verificar subdirectorios train y test
        train_dir = os.path.join(class_dir, 'train')
        test_dir = os.path.join(class_dir, 'test')
        
        # Contar archivos en train
        train_count = 0
        train_size = 0
        if os.path.exists(train_dir):
            train_files = os.listdir(train_dir)
            for file in train_files:
                file_path = os.path.join(train_dir, file)
                if os.path.isfile(file_path):
                    _, ext = os.path.splitext(file)
                    results['file_extensions'][ext] += 1
                    
                    if ext in valid_extensions:
                        train_count += 1
                        train_size += os.path.getsize(file_path)
                    else:
                        results['invalid_files'].append(file_path)
            
            results['directory_structure'][class_name]['train'] = train_files
        
        # Contar archivos en test
        test_count = 0
        test_size = 0
        if os.path.exists(test_dir):
            test_files = os.listdir(test_dir)
            for file in test_files:
                file_path = os.path.join(test_dir, file)
                if os.path.isfile(file_path):
                    _, ext = os.path.splitext(file)
                    results['file_extensions'][ext] += 1
                    
                    if ext in valid_extensions:
                        test_count += 1
                        test_size += os.path.getsize(file_path)
                    else:
                        results['invalid_files'].append(file_path)
            
            results['directory_structure'][class_name]['test'] = test_files
        
        results['train_counts'][class_name] = train_count
        results['test_counts'][class_name] = test_count
        results['total_counts'][class_name] = train_count + test_count
        results['train_sizes'][class_name] = train_size
        results['test_sizes'][class_name] = test_size
        results['total_sizes'][class_name] = train_size + test_size
        
        print(f"{class_name}:")
        print(f"  Train: {train_count} imágenes ({format_size(train_size)})")
        print(f"  Test:  {test_count} imágenes ({format_size(test_size)})")
        print(f"  Total: {train_count + test_count} imágenes ({format_size(train_size + test_size)})")
        print()
    
    return results

def print_summary_statistics(results):
    """Imprime estadísticas resumidas del dataset"""
    print("=" * 60)
    print("RESUMEN ESTADÍSTICAS DEL DATASET")
    print("=" * 60)
    
    total_train = sum(results['train_counts'].values())
    total_test = sum(results['test_counts'].values())
    total_images = total_train + total_test
    
    total_train_size = sum(results['train_sizes'].values())
    total_test_size = sum(results['test_sizes'].values())
    total_dataset_size = total_train_size + total_test_size
    
    print(f"Total de clases: {len(results['classes'])}")
    print(f"Total imágenes de entrenamiento: {total_train} ({format_size(total_train_size)})")
    print(f"Total imágenes de prueba: {total_test} ({format_size(total_test_size)})")
    print(f"Total de imágenes: {total_images} ({format_size(total_dataset_size)})")
    print()
    
    print("Distribución por clase:")
    for class_name in results['classes']:
        train_count = results['train_counts'][class_name]
        test_count = results['test_counts'][class_name]
        total_count = results['total_counts'][class_name]
        train_size = results['train_sizes'][class_name]
        test_size = results['test_sizes'][class_name]
        total_size = results['total_sizes'][class_name]
        
        train_pct = (train_count / total_train * 100) if total_train > 0 else 0
        test_pct = (test_count / total_test * 100) if total_test > 0 else 0
        total_pct = (total_count / total_images * 100) if total_images > 0 else 0
        size_pct = (total_size / total_dataset_size * 100) if total_dataset_size > 0 else 0
        
        print(f"  {class_name}:")
        print(f"    Train: {train_count:3d} ({train_pct:5.1f}%) - {format_size(train_size)}")
        print(f"    Test:  {test_count:3d} ({test_pct:5.1f}%) - {format_size(test_size)}")
        print(f"    Total: {total_count:3d} ({total_pct:5.1f}%) - {format_size(total_size)} ({size_pct:5.1f}% del dataset)")
        print()
    
    print("Extensiones de archivos encontradas:")
    for ext, count in results['file_extensions'].most_common():
        print(f"  {ext}: {count} archivos")
    
    if results['invalid_files']:
        print(f"\nArchivos con extensiones no válidas ({len(results['invalid_files'])}):")
        for file_path in results['invalid_files'][:10]:  # Mostrar solo los primeros 10
            print(f"  {file_path}")
        if len(results['invalid_files']) > 10:
            print(f"  ... y {len(results['invalid_files']) - 10} más")

def create_visualizations(results, output_dir="dataset-reports/plots"):
    """Crea visualizaciones de la distribución del dataset"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Configurar el estilo
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Gráfico de barras para train vs test por clase
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    classes = results['classes']
    train_counts = [results['train_counts'][c] for c in classes]
    test_counts = [results['test_counts'][c] for c in classes]
    
    x = range(len(classes))
    width = 0.35
    
    ax1.bar([i - width/2 for i in x], train_counts, width, label='Train', alpha=0.8)
    ax1.bar([i + width/2 for i in x], test_counts, width, label='Test', alpha=0.8)
    ax1.set_xlabel('Clases')
    ax1.set_ylabel('Número de imágenes')
    ax1.set_title('Distribución de imágenes por clase (Train vs Test)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(classes, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Gráfico de pastel para distribución total por clase
    total_counts = [results['total_counts'][c] for c in classes]
    colors = sns.color_palette("husl", len(classes))
    
    wedges, texts, autotexts = ax2.pie(total_counts, labels=classes, autopct='%1.1f%%', 
                                       colors=colors, startangle=90)
    ax2.set_title('Distribución total de imágenes por clase')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dataset_distribution.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Gráfico de balance del dataset
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Calcular ratio train/test para cada clase
    ratios = []
    for class_name in classes:
        train_count = results['train_counts'][class_name]
        test_count = results['test_counts'][class_name]
        ratio = train_count / test_count if test_count > 0 else float('inf')
        ratios.append(ratio)
    
    bars = ax.bar(classes, ratios, alpha=0.7, color=colors)
    ax.set_xlabel('Clases')
    ax.set_ylabel('Ratio Train/Test')
    ax.set_title('Balance Train/Test por clase')
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    
    # Añadir línea de referencia para ratio ideal (aproximadamente 4:1)
    ax.axhline(y=4, color='red', linestyle='--', alpha=0.7, label='Ratio ideal (4:1)')
    ax.legend()
    
    # Añadir valores en las barras
    for bar, ratio in zip(bars, ratios):
        height = bar.get_height()
        if height != float('inf'):
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{ratio:.1f}', ha='center', va='bottom')
        else:
            ax.text(bar.get_x() + bar.get_width()/2., 1,
                   'inf', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dataset_balance.png'), dpi=300, bbox_inches='tight')
    plt.show()

def save_dataset_report(results, output_file="dataset-reports/dataset_report.txt"):
    """Guarda un reporte detallado del dataset en un archivo"""
    # Crear el directorio si no existe
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("REPORTE DE ANÁLISIS DEL DATASET\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Fecha de análisis: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("ESTADÍSTICAS GENERALES:\n")
        f.write("-" * 25 + "\n")
        f.write(f"Número de clases: {len(results['classes'])}\n")
        f.write(f"Clases: {', '.join(results['classes'])}\n")
        
        total_train_size = sum(results['train_sizes'].values())
        total_test_size = sum(results['test_sizes'].values())
        total_dataset_size = total_train_size + total_test_size
        
        f.write(f"Total imágenes entrenamiento: {sum(results['train_counts'].values())} ({format_size(total_train_size)})\n")
        f.write(f"Total imágenes prueba: {sum(results['test_counts'].values())} ({format_size(total_test_size)})\n")
        f.write(f"Total imágenes: {sum(results['total_counts'].values())} ({format_size(total_dataset_size)})\n\n")
        
        f.write("DISTRIBUCIÓN POR CLASE:\n")
        f.write("-" * 25 + "\n")
        for class_name in results['classes']:
            train_size = results['train_sizes'][class_name]
            test_size = results['test_sizes'][class_name]
            total_size = results['total_sizes'][class_name]
            
            f.write(f"{class_name}:\n")
            f.write(f"  Train: {results['train_counts'][class_name]} ({format_size(train_size)})\n")
            f.write(f"  Test: {results['test_counts'][class_name]} ({format_size(test_size)})\n")
            f.write(f"  Total: {results['total_counts'][class_name]} ({format_size(total_size)})\n\n")
        
        f.write("EXTENSIONES DE ARCHIVOS:\n")
        f.write("-" * 25 + "\n")
        for ext, count in results['file_extensions'].most_common():
            f.write(f"{ext}: {count} archivos\n")
        
        if results['invalid_files']:
            f.write(f"\nARCHIVOS INVÁLIDOS ({len(results['invalid_files'])}):\n")
            f.write("-" * 25 + "\n")
            for file_path in results['invalid_files']:
                f.write(f"{file_path}\n")

def main():
    """Función principal"""
    data_dir = "data"
    
    print("Analizando dataset...")
    print("=" * 60)
    
    # Verificar que el directorio existe
    if not os.path.exists(data_dir):
        print(f"Error: El directorio '{data_dir}' no existe.")
        print("Asegúrate de ejecutar este script desde el directorio raíz del proyecto.")
        return
    
    # Analizar la estructura del dataset
    results = analyze_dataset_structure(data_dir)
    
    # Mostrar estadísticas
    print_summary_statistics(results)
    
    # Crear visualizaciones
    print("\nCreando visualizaciones...")
    try:
        create_visualizations(results)
        print("✓ Visualizaciones guardadas en el directorio 'dataset-reports/plots/'")
    except Exception as e:
        print(f"Error al crear visualizaciones: {e}")
    
    # Guardar reporte
    try:
        save_dataset_report(results)
        print("✓ Reporte guardado como 'dataset-reports/dataset_report.txt'")
    except Exception as e:
        print(f"Error al guardar reporte: {e}")
    
    print("\nAnálisis completado.")

if __name__ == "__main__":
    main()
