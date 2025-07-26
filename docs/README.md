# Índice de Documentación

Esta carpeta contiene documentación técnica detallada sobre los componentes principales del proyecto de clasificación de imágenes "Modelo Entre Diagonales".

## 📚 Documentos Disponibles

### 🏛️ [Arquitecturas de Modelos](arquitecturas.md)
Explicación detallada de las cuatro arquitecturas disponibles:
- **EfficientNetB3**: Modelo balanceado de Google
- **MobileNetV2**: Optimizado para dispositivos móviles
- **ResNeSt50**: Basado en ResNet con mecanismos de atención
- **ConvNeXt-Tiny**: Arquitectura moderna con ideas de Transformers

**Incluye**: Comparación técnica, casos de uso, ventajas y desventajas.

### 🧠 [Funciones de Activación](funciones_activacion.md)
Conceptos fundamentales sobre funciones de activación:
- **ReLU**: Para capas ocultas
- **Softmax**: Para clasificación multiclase
- **Sigmoid**: Contexto y comparaciones

**Incluye**: Fórmulas matemáticas, ejemplos de código, análisis visual.

### 🔧 [Keras y TensorFlow](keras_tensorflow.md)
Guía completa de los frameworks utilizados:
- **Tensores**: Estructuras de datos básicas
- **Capas**: Bloques de construcción de redes
- **Modelos**: Arquitecturas y compilación
- **Entrenamiento**: Pipeline completo

**Incluye**: Ejemplos prácticos, debugging, mejores prácticas.

### ⚙️ [Variables y Configuraciones](variables_configuracion.md)
Documentación de todas las variables del proyecto:
- **Configuraciones globales**: Directorios y parámetros
- **Variables de modelo**: Arquitectura y hiperparámetros
- **Variables de entrenamiento**: Pipeline y optimización
- **Variables de predicción**: Inferencia y preprocesamiento

**Incluye**: Explicaciones prácticas, recomendaciones de modificación.

### 📊 [Métricas y Evaluación](metricas_evaluacion.md)
Explicación completa del sistema de evaluación:
- **Accuracy**: Precisión general
- **Loss**: Función de pérdida
- **F1-Score**: Balance precision/recall
- **Matriz de confusión**: Análisis detallado de errores

**Incluye**: Interpretación práctica, ejemplos de cálculo, casos de uso.

### 🖼️ [Preprocesamiento de Imágenes](preprocesamiento.md)
Técnicas de preparación de datos:
- **Normalización**: Escalado de píxeles
- **Redimensionamiento**: Tamaños de entrada
- **Aumentación de datos**: Técnicas de augmentación
- **Pipeline optimizado**: Mejores prácticas

**Incluye**: Ejemplos visuales, código de implementación, análisis de efectos.

### 📊 Análisis de Dataset
Herramientas para evaluar la calidad y distribución de los datos:
- **Script de análisis**: `analyze_dataset.py` para estadísticas completas
- **Reportes automáticos**: Generación de informes detallados
- **Visualizaciones**: Gráficos de distribución y balance
- **Validación de estructura**: Verificación de formato y organización

**Incluye**: Comando de uso, interpretación de resultados, mejores prácticas.

---

## 🎯 Guías de Lectura Recomendadas

### Para Principiantes:
1. **[Keras y TensorFlow](keras_tensorflow.md)** - Conceptos básicos
2. **[Funciones de Activación](funciones_activacion.md)** - Fundamentos matemáticos
3. **[Preprocesamiento](preprocesamiento.md)** - Preparación de datos
4. **[Análisis de Dataset](../README.md#-análisis-del-dataset)** - Evaluación de datos inicial

### Para Usuarios Intermedios:
1. **[Análisis de Dataset](../README.md#-análisis-del-dataset)** - Validación de datos
2. **[Variables y Configuraciones](variables_configuracion.md)** - Personalización
3. **[Arquitecturas](arquitecturas.md)** - Selección de modelos
4. **[Métricas](metricas_evaluacion.md)** - Evaluación de resultados

### Para Desarrollo Avanzado:
1. **[Arquitecturas](arquitecturas.md)** - Implementación de nuevos modelos
2. **[Variables y Configuraciones](variables_configuracion.md)** - Optimización avanzada
3. **[Preprocesamiento](preprocesamiento.md)** - Pipeline personalizado

---

## 🔍 Búsqueda Rápida por Temas

### Configuración y Setup:
- Variables de directorio → [Variables y Configuraciones](variables_configuracion.md#rutas-y-directorios)
- Instalación de dependencias → [README principal](../README.md#instalación)
- Configuración de GPU → [Keras y TensorFlow](keras_tensorflow.md#gestión-de-memoria)
- Análisis de dataset → [README principal](../README.md#-análisis-del-dataset)

### Entrenamiento:
- Análisis previo de datos → [README principal](../README.md#-análisis-del-dataset)
- Selección de modelo → [Arquitecturas](arquitecturas.md#comparación-de-modelos)
- Hiperparámetros → [Variables y Configuraciones](variables_configuracion.md#variables-de-modelo)
- Aumentación de datos → [Preprocesamiento](preprocesamiento.md#aumentación-de-datos)

### Evaluación:
- Interpretación de métricas → [Métricas](metricas_evaluacion.md#interpretación-práctica)
- Análisis de errores → [Métricas](metricas_evaluacion.md#matriz-de-confusión)
- Overfitting/Underfitting → [Métricas](metricas_evaluacion.md#training-vs-validation)

### Predicción:
- Carga de modelos → [Keras y TensorFlow](keras_tensorflow.md#predicciones)
- Preprocesamiento de nuevas imágenes → [Preprocesamiento](preprocesamiento.md#pipeline-de-datos-optimizado)
- Interpretación de resultados → [Métricas](metricas_evaluacion.md#métricas-de-confianza)

---

## 🛠️ Casos de Uso Comunes

### "¿Mi dataset está balanceado?"
1. Ejecutar [análisis de dataset](../README.md#-análisis-del-dataset)
2. Revisar [reportes automáticos](../README.md#-archivos-generados)
3. Interpretar [gráficos de distribución](../README.md#-visualizaciones-automáticas)

### "Mi modelo no converge"
1. Revisar [learning rate](variables_configuracion.md#learning-rate-tasa-de-aprendizaje)
2. Verificar [normalización](preprocesamiento.md#normalización-de-píxeles)
3. Analizar [curvas de entrenamiento](metricas_evaluacion.md#training-vs-validation)

### "Quiero cambiar el tamaño de imagen"
1. Modificar [configuración del modelo](variables_configuracion.md#configuraciones-específicas-por-modelo)
2. Entender [impacto en arquitectura](arquitecturas.md#características-técnicas)
3. Ajustar [preprocesamiento](preprocesamiento.md#redimensionamiento-automático)

### "Mi dataset es pequeño"
1. Aumentar [augmentación](preprocesamiento.md#aumentación-de-datos)
2. Ajustar [regularización](variables_configuracion.md#dropout-rate)
3. Usar [transfer learning](arquitecturas.md#transfer-learning)

### "Quiero agregar una nueva arquitectura"
1. Estudiar [patrón de implementación](arquitecturas.md#ejemplo-de-implementación)
2. Configurar [variables del modelo](variables_configuracion.md#variables-de-modelo)
3. Definir [función de activación](funciones_activacion.md#implementación-en-el-proyecto)

---

## 🔗 Enlaces Útiles

- **[README Principal](../README.md)**: Guía de inicio rápido
- **[Código Fuente](../models.py)**: Implementación de modelos
- **[Configuración](../config.py)**: Parámetros del proyecto
- **[Utilidades](../utils.py)**: Funciones auxiliares

---

*Última actualización: Julio 2025*
