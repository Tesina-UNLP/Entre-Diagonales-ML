# Funciones de Activación en Deep Learning

Las funciones de activación son componentes fundamentales en las redes neuronales que determinan la salida de cada neurona. Este documento explica las funciones utilizadas en el proyecto.

## 🧠 ¿Qué es una Función de Activación?

Una función de activación decide si una neurona debe activarse o no, introduciendo **no-linealidad** en la red neuronal.

### Sin Activación (Lineal):
```
salida = entrada1 * peso1 + entrada2 * peso2 + bias
```

### Con Activación:
```
salida = activacion(entrada1 * peso1 + entrada2 * peso2 + bias)
```

## 📊 Funciones Utilizadas en el Proyecto

### 1. ReLU (Rectified Linear Unit)

**Fórmula**: `f(x) = max(0, x)`

```python
# Implementación en Keras
layers.Dense(64, activation="relu")
```

#### Características:
- **Rango**: [0, +∞)
- **Derivada**: 1 si x > 0, 0 si x ≤ 0
- **Uso**: Capas ocultas

#### Ventajas:
- ✅ Computacionalmente eficiente
- ✅ Evita el problema del gradiente desvaneciente
- ✅ Convergencia rápida

#### Desventajas:
- ❌ "Dying ReLU" - neuronas pueden "morir"
- ❌ No simétrica alrededor del cero

#### Ejemplo Práctico:
```python
import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0, x)

x = np.linspace(-5, 5, 100)
y = relu(x)

plt.plot(x, y)
plt.title('Función ReLU')
plt.xlabel('Entrada')
plt.ylabel('Salida')
plt.grid(True)
plt.show()
```

---

### 2. Softmax

**Fórmula**: `f(xi) = e^xi / Σ(e^xj)`

```python
# Implementación en Keras
layers.Dense(num_classes, activation="softmax")
```

#### Características:
- **Rango**: (0, 1)
- **Suma total**: 1.0
- **Uso**: Capa de salida para clasificación multiclase

#### Propiedades:
- ✅ Convierte logits en probabilidades
- ✅ Resalta la clase con mayor puntuación
- ✅ Diferenciable

#### Ejemplo Práctico:
```python
import numpy as np

def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Estabilidad numérica
    return exp_x / np.sum(exp_x)

# Ejemplo con 3 clases
logits = np.array([2.0, 1.0, 0.1])
probabilidades = softmax(logits)

print(f"Logits: {logits}")
print(f"Probabilidades: {probabilidades}")
print(f"Suma: {np.sum(probabilidades)}")

# Salida esperada:
# Logits: [2.  1.  0.1]
# Probabilidades: [0.659 0.242 0.099]
# Suma: 1.0
```

---

### 3. Sigmoid (en contexto de Dropout)

Aunque no se usa directamente como activación en nuestros modelos, es importante entenderla:

**Fórmula**: `f(x) = 1 / (1 + e^-x)`

```python
# No usado directamente, pero importante conceptualmente
layers.Dense(1, activation="sigmoid")  # Para clasificación binaria
```

#### Características:
- **Rango**: (0, 1)
- **Forma**: Curva S
- **Uso**: Clasificación binaria, gates en LSTM

## 🔧 Implementación en el Proyecto

### En EfficientNet:
```python
# Capas internas: ReLU (automático en EfficientNet)
base_model = EfficientNetB3(...)

# Capa final: Softmax
outputs = layers.Dense(num_classes, activation="softmax")(x)
```

### En MobileNet:
```python
# Capa intermedia: ReLU explícita
x = layers.Dense(64, activation="relu")(x)

# Capa final: Softmax
outputs = layers.Dense(num_classes, activation="softmax")(x)
```

### En ResNeSt:
```python
# Capa intermedia: ReLU
layers.Dense(256, activation='relu'),

# Capa final: Softmax
layers.Dense(num_classes, activation='softmax')
```

### En ConvNeXt:
```python
# Capas internas: GELU (automático en ConvNeXt)
base_model = ConvNeXtTiny(...)

# Capa intermedia: ReLU opcional
x = layers.Dense(128, activation="relu")(x)

# Capa final: Softmax
outputs = layers.Dense(num_classes, activation="softmax")(x)
```

## 📈 Análisis de Comportamiento

### Comparación Visual:

```python
import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax_example(x):
    # Ejemplo con x como array de 3 elementos
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)

x = np.linspace(-5, 5, 100)

plt.figure(figsize=(15, 5))

# ReLU
plt.subplot(1, 3, 1)
plt.plot(x, relu(x), 'r-', linewidth=2)
plt.title('ReLU')
plt.grid(True)

# Sigmoid
plt.subplot(1, 3, 2)
plt.plot(x, sigmoid(x), 'g-', linewidth=2)
plt.title('Sigmoid')
plt.grid(True)

# Softmax es para vectores, no escalares
plt.subplot(1, 3, 3)
x_vec = np.array([2, 1, 0.1])
y_vec = softmax_example(x_vec)
plt.bar(['Clase 1', 'Clase 2', 'Clase 3'], y_vec)
plt.title('Softmax (Ejemplo)')
plt.ylabel('Probabilidad')

plt.tight_layout()
plt.show()
```

## 🎯 Casos de Uso Específicos

### 1. Clasificación Multiclase (Nuestro caso):
```python
# CORRECTO: Softmax en la salida
outputs = layers.Dense(num_classes, activation="softmax")(x)

# INCORRECTO: ReLU en la salida
# outputs = layers.Dense(num_classes, activation="relu")(x)
```

### 2. Capas Intermedias:
```python
# CORRECTO: ReLU para capas ocultas
x = layers.Dense(64, activation="relu")(x)

# EVITAR: Sigmoid en capas profundas (gradiente desvaneciente)
# x = layers.Dense(64, activation="sigmoid")(x)
```

## 🔬 Experimentos y Observaciones

### Efecto de Diferentes Activaciones:

```python
# Experimentar con diferentes activaciones en capas intermedias
activaciones = ['relu', 'gelu', 'swish', 'elu']

for activacion in activaciones:
    model = tf.keras.Sequential([
        layers.Dense(64, activation=activacion),
        layers.Dense(32, activation=activacion),
        layers.Dense(num_classes, activation='softmax')
    ])
    # Entrenar y comparar resultados
```

### Análisis de Gradientes:
```python
# Verificar que los gradientes fluyen correctamente
with tf.GradientTape() as tape:
    predictions = model(x_batch)
    loss = loss_fn(y_batch, predictions)

gradients = tape.gradient(loss, model.trainable_variables)

# Verificar gradientes no son NaN o muy pequeños
for i, grad in enumerate(gradients):
    if grad is not None:
        print(f"Capa {i}: grad_norm = {tf.norm(grad):.6f}")
```

## 💡 Consejos Prácticos

### 1. Selección de Activación:
- **Capas ocultas**: ReLU (por defecto)
- **Clasificación multiclase**: Softmax
- **Clasificación binaria**: Sigmoid
- **Regresión**: Linear (sin activación)

### 2. Debugging Activaciones:
```python
# Monitorear activaciones durante entrenamiento
class ActivationMonitor(tf.keras.callbacks.Callback):
    def on_batch_end(self, batch, logs=None):
        # Obtener activaciones de capas intermedias
        layer_outputs = [layer.output for layer in self.model.layers[1:]]
        activation_model = tf.keras.Model(self.model.input, layer_outputs)
        activations = activation_model.predict(x_sample)
        
        # Verificar distribución de activaciones
        for i, act in enumerate(activations):
            print(f"Capa {i}: mean={np.mean(act):.3f}, std={np.std(act):.3f}")
```

### 3. Problemas Comunes:
```python
# Problema: Softmax con valores muy grandes
logits = [1000, 999, 998]  # Causará overflow

# Solución: Normalización
logits_norm = logits - np.max(logits)  # [-1, -2, -3]
probs = softmax(logits_norm)  # Estable numéricamente
```
