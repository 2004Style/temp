# 🤟 Reconocedor de Lenguaje de Señas ASL

Sistema de reconocimiento de lenguaje de señas americano (ASL) usando Deep Learning con TensorFlow/Keras.

## 📋 Descripción

Este proyecto entrena una red neuronal convolucional (CNN) para reconocer gestos de manos del alfabeto de lenguaje de señas americano. El modelo puede distinguir entre 29 clases diferentes incluyendo las 26 letras del alfabeto más gestos especiales (`del`, `nothing`, `space`).

## 🚀 Características

- **Modelo CNN optimizado** con aumento de datos para mejor generalización
- **Creación automática de carpetas** de logs y exportación
- **Exportación a TensorFlow.js** para uso en aplicaciones web
- **Interfaz web interactiva** para probar el modelo
- **Visualizaciones** del proceso de entrenamiento y datos

## 📂 Estructura del Proyecto

```
temp/
├── index.py                    # Script principal de entrenamiento
├── exportar_tensorflowjs.py    # Script para exportar a TensorFlow.js
├── uso_modelo.py               # Ejemplos de uso del modelo
├── reconocedor_asl.html        # Aplicación web interactiva
├── requirements.txt            # Dependencias del proyecto
├── asl_alphabet_train/         # Dataset de imágenes ASL
│   ├── A/                      # Imágenes de la letra A
│   ├── B/                      # Imágenes de la letra B
│   └── ...                     # Otras letras y gestos
├── logs/                       # Logs de TensorBoard (se crea automáticamente)
├── modelo_exportado/           # Modelo exportado para web (se crea automáticamente)
└── web/                        # Archivos web adicionales
```

## 🛠️ Instalación

1. **Clonar o descargar el proyecto**

2. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Asegurar que tienes el dataset ASL:**
   - El dataset debe estar en la carpeta `asl_alphabet_train/`
   - Cada subcarpeta representa una clase (A, B, C, ..., Z, del, nothing, space)

## 📖 Uso

### 1. Entrenar el Modelo

Ejecuta el script principal para entrenar el modelo:

```bash
python index.py
```

Este script:
- ✅ Carga y preprocesa las imágenes del dataset
- ✅ Crea automáticamente las carpetas de logs
- ✅ Configura aumento de datos para mejor generalización
- ✅ Entrena un modelo CNN optimizado
- ✅ Guarda el modelo como `modelo_asl_augmented.h5`
- ✅ Genera gráficas de entrenamiento
- ✅ Exporta automáticamente a TensorFlow.js

### 2. Exportar a TensorFlow.js (Opcional)

Si necesitas exportar el modelo por separado:

```bash
python exportar_tensorflowjs.py
```

### 3. Usar el Modelo Entrenado

#### En Python:
```python
import numpy as np
import tensorflow as tf
from uso_modelo import predecir_gesto

# Cargar modelo
modelo = tf.keras.models.load_model('modelo_asl_augmented.h5')

# Hacer predicción
clase, probabilidad, todas_prob = predecir_gesto('ruta/imagen.jpg')
print(f"Predicción: {clase} (Confianza: {probabilidad:.2f})")
```

#### En Aplicación Web:
1. Abre `reconocedor_asl.html` en tu navegador
2. Arrastra una imagen o selecciona un archivo
3. Ve la predicción en tiempo real

## 🧠 Arquitectura del Modelo

```
Entrada: Imagen 100x100x1 (escala de grises)
│
├── Conv2D(32 filtros, 3x3) + ReLU
├── MaxPooling2D(2x2)
├── Conv2D(64 filtros, 3x3) + ReLU
├── MaxPooling2D(2x2)
├── Conv2D(128 filtros, 3x3) + ReLU
├── MaxPooling2D(2x2)
├── Dropout(0.5)
├── Flatten()
├── Dense(128) + ReLU
├── Dropout(0.3)
└── Dense(29) + Softmax
│
Salida: Probabilidades para 29 clases
```

## 📊 Aumento de Datos

El modelo usa las siguientes transformaciones para aumentar el dataset:

- **Rotación:** ±30 grados
- **Desplazamiento:** ±20% horizontal y vertical
- **Cizalla:** ±15 grados
- **Zoom:** 70% - 140%
- **Volteos:** Horizontal y vertical

## 🎯 Clases Reconocidas

El modelo puede reconocer **29 clases:**

**Letras del alfabeto:** A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z

**Gestos especiales:**
- `del` - Borrar
- `nothing` - Sin gesto
- `space` - Espacio

## 📈 Monitoreo con TensorBoard

Para visualizar el entrenamiento en tiempo real:

```bash
tensorboard --logdir logs/asl_cnn_augmented
```

Luego abre http://localhost:6006 en tu navegador.

## 🌐 Uso en Aplicaciones Web

El modelo exportado se puede usar en cualquier aplicación web:

```javascript
// Cargar modelo
const modelo = await tf.loadLayersModel('modelo_exportado/modelo_asl_js/model.json');

// Hacer predicción
const prediccion = await modelo.predict(tensorImagen);
const clasePredicha = prediccion.argMax(-1).dataSync()[0];
```

## 📋 Requisitos del Sistema

- **Python:** 3.7+
- **TensorFlow:** 2.x
- **Memoria RAM:** Mínimo 8GB recomendado
- **GPU:** Opcional pero recomendada para entrenamiento más rápido

## 🚨 Solución de Problemas

### Error de memoria durante entrenamiento:
- Reduce el `batch_size` de 32 a 16 o 8
- Reduce el número de `epochs`

### Modelo no carga en web:
- Verifica que la ruta al modelo sea correcta
- Asegúrate de servir los archivos desde un servidor HTTP

### Baja precisión del modelo:
- Aumenta el número de épocas
- Verifica la calidad del dataset
- Ajusta los hiperparámetros

## 📄 Archivos Generados

Después del entrenamiento tendrás:

- `modelo_asl_augmented.h5` - Modelo de Keras
- `modelo_exportado/modelo_asl_js/` - Modelo para TensorFlow.js
- `logs/asl_cnn_augmented/` - Logs de TensorBoard

## 🤝 Contribuciones

¡Las contribuciones son bienvenidas! Puedes:

1. Mejorar la arquitectura del modelo
2. Añadir más clases o gestos
3. Optimizar el preprocesamiento
4. Mejorar la interfaz web

## 📝 Notas Importantes

- **Calidad de imágenes:** El modelo funciona mejor con imágenes claras y bien iluminadas
- **Posición de manos:** Mantén las manos centradas en la imagen
- **Fondo:** Preferiblemente usa fondos uniformes
- **Tamaño:** Las imágenes se redimensionan automáticamente a 100x100

---

¡Disfruta reconociendo lenguaje de señas con IA! 🤖✋
