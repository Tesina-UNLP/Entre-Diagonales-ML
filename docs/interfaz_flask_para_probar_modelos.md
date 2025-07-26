# 🏛️ Aplicación Flask - Clasificador de Arquitectura de La Plata

Una aplicación web sencilla que permite subir imágenes de edificios emblemáticos de La Plata y obtener predicciones del modelo de clasificación.

## 🚀 Características

- **Interfaz web intuitiva**: Arrastra y suelta imágenes o selecciona archivos
- **Predicciones en tiempo real**: Obtén las top 5 predicciones con porcentajes de confianza
- **API REST**: Endpoint para integración programática
- **Soporte múltiple**: Compatible con diferentes arquitecturas de modelos (EfficientNet, MobileNet, ConvNeXt)

## 🏗️ Edificios que puede identificar

1. **Casa Curutchet** - Obra maestra de Le Corbusier
2. **Catedral** - Catedral de La Plata
3. **Dardo Rocha** - Estación y Centro Cultural
4. **Municipalidad** - Palacio Municipal
5. **Museo de Ciencias Naturales** - Museo de La Plata
6. **Teatro Argentino** - Teatro principal de la ciudad

## 📋 Requisitos

Asegúrate de tener Python 3.8+ instalado y ejecuta:

```bash
pip install -r requirements.txt
```

### Dependencias principales

- Flask 2.0+
- TensorFlow 2.10+
- Pillow (PIL)
- NumPy

## 🛠️ Instalación y Uso

### 1. Clonar y preparar el entorno

```bash
# Navegar al directorio del proyecto
cd modelo-entre-diagonales

# Instalar dependencias
pip install -r requirements.txt
```

### 2. Entrenar un modelo (si no tienes uno)

Si no tienes un modelo entrenado en la carpeta `models/`, puedes entrenar uno rápido:

```bash
python train_quick.py
```

Este script creará un modelo básico con las imágenes que tengas en el directorio `data/`.

### 3. Ejecutar la aplicación

```bash
python app.py
```

La aplicación estará disponible en: **http://localhost:5000**

## 🌐 Uso de la Aplicación

### Interfaz Web

1. **Acceder**: Abre tu navegador en `http://localhost:5000`
2. **Subir imagen**: 
   - Arrastra una imagen al área designada, o
   - Haz clic en "Seleccionar Imagen"
3. **Analizar**: Presiona "Analizar Imagen"
4. **Ver resultados**: Obtén las top 5 predicciones con porcentajes

### API REST

#### Endpoint de predicción

```http
POST /api/predict
Content-Type: multipart/form-data

file: [imagen.jpg]
```

#### Respuesta exitosa

```json
{
  "success": true,
  "model_name": "efficientnetb3",
  "predictions": [
    {
      "rank": 1,
      "class": "Casa Curutchet",
      "confidence": 85.3
    },
    {
      "rank": 2,
      "class": "Catedral",
      "confidence": 12.7
    }
  ]
}
```

#### Endpoint de salud

```http
GET /health
```

## 📁 Estructura de Archivos

```
modelo-entre-diagonales/
├── app.py                 # Aplicación Flask principal
├── train_quick.py         # Script de entrenamiento rápido
├── templates/             # Templates HTML
│   ├── index.html         # Página principal
│   └── results.html       # Página de resultados
├── models/                # Modelos entrenados (.h5)
├── uploads/               # Directorio temporal para uploads
├── data/                  # Dataset de imágenes
└── requirements.txt       # Dependencias
```

## 🔧 Configuración

### Modificar clases

Si quieres cambiar las clases que identifica el modelo, edita el diccionario `CLASS_NAMES` en `app.py`:

```python
CLASS_NAMES = {
    0: 'Tu Clase 1',
    1: 'Tu Clase 2',
    # ...
}
```

### Configuración de modelos

El sistema detecta automáticamente el tipo de modelo por el nombre del archivo:

- `*mobilenet*` → MobileNetV2
- `*efficientnet*` → EfficientNetB3  
- `*convnext*` → ConvNeXt Tiny

### Variables de entorno

Puedes configurar:

```bash
export FLASK_ENV=development  # Para modo desarrollo
export FLASK_DEBUG=1          # Para debug
```

## 🚨 Resolución de Problemas

### Error: "No se encontraron modelos entrenados"
- Asegúrate de tener archivos `.h5` en la carpeta `models/`
- O ejecuta `python train_quick.py` para entrenar uno básico

### Error: "Modelo no cargado"
- Verifica que el modelo sea compatible con TensorFlow
- Revisa que todas las dependencias estén instaladas

### Error al subir imagen
- Verifica que el archivo sea una imagen válida (JPG, PNG, GIF)
- Comprueba que el tamaño no exceda 16MB

### Errores de importación

```bash
# Reinstalar dependencias
pip install --upgrade -r requirements.txt

# Verificar instalación de TensorFlow
python -c "import tensorflow; print(tensorflow.__version__)"
```

## 🔍 Testing

### Probar con imágenes de ejemplo

Usa las imágenes en `test/` para probar rápidamente:

```bash
# Copiar imagen de test a uploads para prueba manual
cp test/catedral.jpg uploads/
```

### Probar API con curl

```bash
curl -X POST -F "file=@test/catedral.jpg" http://localhost:5000/api/predict
```

## 📊 Monitoreo

La aplicación incluye:

- **Endpoint de salud**: `/health` para verificar estado
- **Logs**: Se muestran en la consola durante ejecución
- **Manejo de errores**: Mensajes descriptivos para debugging

## 🎨 Personalización

### Cambiar estilo visual

Edita los archivos en `templates/` para modificar:

- Colores y temas
- Layout y estructura
- Texto e idioma

### Añadir funcionalidades

- Historial de predicciones
- Guardado de resultados
- Comparación de modelos
- Batch processing

## 📝 Notas Importantes

- **Seguridad**: Esta es una aplicación de desarrollo/demo
- **Performance**: El modelo se carga en memoria al iniciar
- **Escalabilidad**: Para producción, considera usar Gunicorn/uWSGI
- **Almacenamiento**: Las imágenes se eliminan tras procesamiento

## 🤝 Contribuir

Para mejorar la aplicación:

1. Fork del repositorio
2. Crear rama feature (`git checkout -b feature/nueva-funcionalidad`)
3. Commit cambios (`git commit -am 'Añadir nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Crear Pull Request

---

¡Disfruta clasificando la arquitectura de La Plata! 🏛️✨
