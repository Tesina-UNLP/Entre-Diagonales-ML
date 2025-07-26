# Métricas y Evaluación de Modelos

Este documento explica las métricas utilizadas para evaluar el rendimiento de los modelos de clasificación de imágenes, con ejemplos prácticos y interpretación.

## 📊 Métricas Principales

### 1. Accuracy (Precisión General)

**Definición**: Porcentaje de predicciones correctas sobre el total.

```python
accuracy = (predicciones_correctas) / (total_predicciones)
```

#### Ejemplo Práctico:
```python
# Dataset con 1000 imágenes de prueba
total_imagenes = 1000
correctas = 850

accuracy = 850 / 1000 = 0.85 = 85%
```

#### Interpretación:
- **85-95%**: Excelente rendimiento
- **70-85%**: Buen rendimiento
- **60-70%**: Rendimiento aceptable
- **<60%**: Rendimiento pobre

**Cuándo usar**: Dataset balanceado con clases similares en cantidad.

**Cuándo NO usar**: Dataset desbalanceado (ej: 95% clase A, 5% clase B).

---

### 2. Loss (Función de Pérdida)

**Definición**: Mide qué tan "equivocado" está el modelo en sus predicciones.

#### Sparse Categorical Crossentropy:
```python
loss = -Σ(y_true * log(y_pred))

# Para clasificación multiclase
# y_true: clase real (0, 1, 2, ...)
# y_pred: probabilidades predichas [0.1, 0.7, 0.2]
```

#### Ejemplo Práctico:
```python
import numpy as np

def sparse_categorical_crossentropy(y_true, y_pred):
    # y_true: [1, 0, 2]  (clases reales)
    # y_pred: [[0.1, 0.8, 0.1], [0.9, 0.05, 0.05], [0.2, 0.3, 0.5]]
    
    losses = []
    for i, true_class in enumerate(y_true):
        predicted_prob = y_pred[i][true_class]
        loss = -np.log(predicted_prob)
        losses.append(loss)
    
    return np.mean(losses)

# Ejemplo de uso
y_true = [1, 0, 2]  # Clases reales
y_pred = [[0.1, 0.8, 0.1],   # Buena predicción para clase 1
          [0.9, 0.05, 0.05],  # Buena predicción para clase 0
          [0.2, 0.3, 0.5]]    # Predicción regular para clase 2

loss = sparse_categorical_crossentropy(y_true, y_pred)
print(f"Loss: {loss:.4f}")  # Menor es mejor
```

#### Interpretación:
- **0.0-0.5**: Excelente (predicciones muy confiables)
- **0.5-1.0**: Bueno
- **1.0-2.0**: Aceptable
- **>2.0**: Pobre (modelo muy inseguro)

---

### 3. F1-Score

**Definición**: Media armónica entre Precision y Recall.

```python
f1 = 2 * (precision * recall) / (precision + recall)
```

#### Componentes:

**Precision**: De las predicciones positivas, ¿cuántas fueron correctas?
```python
precision = true_positives / (true_positives + false_positives)
```

**Recall**: De los casos positivos reales, ¿cuántos detectamos?
```python
recall = true_positives / (true_positives + false_negatives)
```

#### Ejemplo Práctico:
```python
# Clasificación de 3 clases: perros, gatos, pájaros
# Matriz de confusión para clase "perros":

#           Predicho
#         P  G  Pa
# Real P [85  3   2]  # 85 perros correctos, 3 confundidos con gatos, 2 con pájaros
#      G [ 4 78   8]  # 4 gatos confundidos con perros
#      Pa[ 1  5  84]  # 1 pájaro confundido con perro

# Para clase "perros":
true_positives = 85    # Perros correctamente identificados
false_positives = 4 + 1 = 5  # Gatos y pájaros identificados como perros
false_negatives = 3 + 2 = 5  # Perros identificados como otras clases

precision_perros = 85 / (85 + 5) = 0.944  # 94.4%
recall_perros = 85 / (85 + 5) = 0.944     # 94.4%
f1_perros = 2 * (0.944 * 0.944) / (0.944 + 0.944) = 0.944  # 94.4%
```

#### F1-Score Macro (Usado en el proyecto):
```python
# Promedio de F1 de todas las clases
f1_macro = (f1_perros + f1_gatos + f1_pajaros) / 3
```

#### Interpretación:
- **0.9-1.0**: Excelente balance precision/recall
- **0.8-0.9**: Muy bueno
- **0.7-0.8**: Bueno
- **<0.7**: Necesita mejora

---

### 4. Matriz de Confusión

**Definición**: Tabla que muestra predicciones vs realidad para cada clase.

#### Ejemplo Visual:
```python
import seaborn as sns
import matplotlib.pyplot as plt

# Matriz de confusión de ejemplo
confusion_matrix = [
    [85,  3,  2],  # Perros: 85 correctos, 3 como gatos, 2 como pájaros
    [ 4, 78,  8],  # Gatos: 4 como perros, 78 correctos, 8 como pájaros
    [ 1,  5, 84]   # Pájaros: 1 como perro, 5 como gatos, 84 correctos
]

plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=['Perros', 'Gatos', 'Pájaros'],
            yticklabels=['Perros', 'Gatos', 'Pájaros'])
plt.title('Matriz de Confusión')
plt.xlabel('Predicción')
plt.ylabel('Clase Real')
plt.show()
```

#### Interpretación:
- **Diagonal principal**: Predicciones correctas (más oscuro = mejor)
- **Fuera de diagonal**: Errores y confusiones
- **Patrones**: Identificar qué clases se confunden más

---

## 📈 Métricas Durante el Entrenamiento

### Training vs Validation

#### Curvas de Aprendizaje:
```python
import matplotlib.pyplot as plt

def plot_training_metrics(history):
    epochs = range(1, len(history.history['accuracy']) + 1)
    
    plt.figure(figsize=(12, 4))
    
    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history.history['accuracy'], 'bo-', label='Training')
    plt.plot(epochs, history.history['val_accuracy'], 'ro-', label='Validation')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history.history['loss'], 'bo-', label='Training')
    plt.plot(epochs, history.history['val_loss'], 'ro-', label='Validation')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
```

#### Interpretación de Curvas:

**Caso Ideal:**
```
Training Accuracy:   ↗↗↗→→→  (sube y se estabiliza)
Validation Accuracy: ↗↗↗→→→  (similar al training)
Training Loss:       ↘↘↘→→→  (baja y se estabiliza)
Validation Loss:     ↘↘↘→→→  (similar al training)
```

**Overfitting:**
```
Training Accuracy:   ↗↗↗↗↗↗  (sigue subiendo)
Validation Accuracy: ↗↗→↘↘↘  (sube y luego baja)
Training Loss:       ↘↘↘↘↘↘  (sigue bajando)
Validation Loss:     ↘↘→↗↗↗  (baja y luego sube)
```

**Underfitting:**
```
Training Accuracy:   ↗→→→→→  (se estanca pronto)
Validation Accuracy: ↗→→→→→  (similar pero bajo)
Training Loss:       ↘→→→→→  (se estanca alto)
Validation Loss:     ↘→→→→→  (similar al training)
```

---

## 🔍 Métricas Específicas por Clase

### Reporte de Clasificación:
```python
from sklearn.metrics import classification_report

def generate_detailed_report(y_true, y_pred, class_names):
    report = classification_report(
        y_true, y_pred, 
        target_names=class_names, 
        output_dict=True
    )
    
    # Imprimir métricas por clase
    for class_name in class_names:
        precision = report[class_name]['precision']
        recall = report[class_name]['recall']
        f1 = report[class_name]['f1-score']
        support = report[class_name]['support']
        
        print(f"\n{class_name}:")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall:    {recall:.3f}")
        print(f"  F1-Score:  {f1:.3f}")
        print(f"  Samples:   {support}")
    
    # Métricas globales
    print(f"\nMacro Average F1: {report['macro avg']['f1-score']:.3f}")
    print(f"Weighted Average F1: {report['weighted avg']['f1-score']:.3f}")
    print(f"Overall Accuracy: {report['accuracy']:.3f}")
```

### Análisis de Errores por Clase:
```python
def analyze_class_performance(confusion_matrix, class_names):
    n_classes = len(class_names)
    
    for i, class_name in enumerate(class_names):
        # Correctos para esta clase
        correct = confusion_matrix[i, i]
        
        # Total de esta clase
        total_class = sum(confusion_matrix[i, :])
        
        # Accuracy de esta clase
        class_accuracy = correct / total_class
        
        # Principales confusiones
        errors = []
        for j in range(n_classes):
            if i != j and confusion_matrix[i, j] > 0:
                errors.append((class_names[j], confusion_matrix[i, j]))
        
        errors.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n{class_name}:")
        print(f"  Accuracy: {class_accuracy:.3f}")
        print(f"  Correctos: {correct}/{total_class}")
        
        if errors:
            print("  Principales confusiones:")
            for confused_class, count in errors[:3]:
                print(f"    → {confused_class}: {count} casos")
```

---

## 🎯 Métricas de Confianza

### Análisis de Probabilidades:
```python
def analyze_prediction_confidence(model, test_dataset, class_names):
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
    
    return all_predictions
```

### Top-K Accuracy:
```python
def top_k_accuracy(y_true, y_pred, k=3):
    """
    Calcula si la clase correcta está entre las top-k predicciones
    """
    correct = 0
    total = len(y_true)
    
    for i in range(total):
        true_class = y_true[i]
        # Obtener las k clases con mayor probabilidad
        top_k_classes = np.argsort(y_pred[i])[-k:]
        
        if true_class in top_k_classes:
            correct += 1
    
    return correct / total

# Ejemplo de uso
top_1_acc = top_k_accuracy(y_true, y_pred, k=1)  # Accuracy normal
top_3_acc = top_k_accuracy(y_true, y_pred, k=3)  # ¿Está en top 3?

print(f"Top-1 Accuracy: {top_1_acc:.3f}")
print(f"Top-3 Accuracy: {top_3_acc:.3f}")
```

---

## 📊 Métricas en el Proyecto

### Variables Guardadas Automáticamente:

#### En `metadata.json`:
```json
{
  "created_at": "20240125_143052",
  "val_accuracy": 0.8542,
  "val_loss": 0.4123,
  "f1_macro": 0.8234,
  "class_names": ["clase1", "clase2", "clase3"],
  "base_model": "efficientnetb3"
}
```

#### En `evaluation_per_class.json`:
```json
{
  "clase1": {
    "precision": 0.8567,
    "recall": 0.8234,
    "f1": 0.8398
  },
  "clase2": {
    "precision": 0.9123,
    "recall": 0.8876,
    "f1": 0.8998
  }
}
```

#### En `experiments/history.csv`:
```csv
timestamp,model_name,accuracy,val_accuracy,val_loss,f1_score,epochs,class_names,path
20240125_143052,efficientnet,0.9123,0.8542,0.4123,0.8234,15,"clase1,clase2,clase3",models/efficientnet_20240125_143052
```

---

## 💡 Interpretación Práctica

### ¿Qué métrica usar cuándo?

**Para comparar modelos:**
- F1-Score macro (balance general)
- Validation accuracy (rendimiento real)

**Para detectar problemas:**
- Training vs Validation curves (overfitting/underfitting)
- Matriz de confusión (confusiones específicas)

**Para producción:**
- Confianza de predicciones
- Top-K accuracy para sugerencias

### Ejemplos de Interpretación:

```python
# Caso 1: Modelo balanceado
metrics = {
    'val_accuracy': 0.85,
    'f1_macro': 0.84,
    'precision_avg': 0.85,
    'recall_avg': 0.83
}
print("✅ Modelo balanceado y confiable")

# Caso 2: Alta precisión, bajo recall
metrics = {
    'val_accuracy': 0.78,
    'f1_macro': 0.72,
    'precision_avg': 0.92,
    'recall_avg': 0.65
}
print("⚠️ Modelo conservador: pocas predicciones pero muy precisas")

# Caso 3: Bajo precision, alto recall
metrics = {
    'val_accuracy': 0.75,
    'f1_macro': 0.71,
    'precision_avg': 0.68,
    'recall_avg': 0.89
}
print("⚠️ Modelo agresivo: detecta mucho pero con falsos positivos")
```
