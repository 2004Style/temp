# === SISTEMA DE RECONOCIMIENTO DE LENGUAJE DE SEÑAS AMERICANO (ASL) ===
# Modelo CNN con aumento de datos para clasificar gestos de manos del alfabeto ASL

import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Crear carpetas necesarias si no existen
os.makedirs('logs/asl_cnn_augmented', exist_ok=True)
os.makedirs('modelo_exportado', exist_ok=True)
print("Carpetas de logs y exportación creadas")

ruta_base = './asl_alphabet_train'  # Ajusta si tu ruta es diferente
TAMANO_IMG = 100

clases = sorted([d for d in os.listdir(ruta_base) if os.path.isdir(os.path.join(ruta_base, d))])
train = []
for idx, clase in enumerate(clases):
    ruta_clase = os.path.join(ruta_base, clase)
    for archivo in os.listdir(ruta_clase):
        if archivo.endswith('.jpg') or archivo.endswith('.png'):
            ruta_img = os.path.join(ruta_clase, archivo)
            img = cv2.imread(ruta_img)
            if img is not None:
                img = cv2.resize(img, (TAMANO_IMG, TAMANO_IMG))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = img.reshape(TAMANO_IMG, TAMANO_IMG, 1)
                train.append((img, idx))  # Tupla (imagen, etiqueta)

# NO crear X ni y como arrays aquí para evitar duplicación de datos y consumo de RAM
datos = {'train': train}
metadatos = {'clases': clases, 'total_imagenes': len(train), 'tamano_img': TAMANO_IMG}

print(f'Total de imágenes cargadas: {len(train)}')
print(f'Clases encontradas: {clases}')


# Mostrar información básica de los datos y metadatos (simulado)
print('Metadatos:')
print(metadatos)
print('Primer ejemplo:')
ejemplo_img, ejemplo_et = datos['train'][0]
print(f"Clase: {metadatos['clases'][ejemplo_et]}")
print(f"Tamaño imagen: {ejemplo_img.shape}")

# Mostrar 5 ejemplos del set usando datos['train'] y metadatos
fig, axs = plt.subplots(1, 5, figsize=(15, 3))
for i in range(5):
    img, etiqueta = datos['train'][i]
    img = img.reshape(metadatos['tamano_img'], metadatos['tamano_img'])
    axs[i].imshow(img, cmap='gray')
    axs[i].set_title(metadatos['clases'][etiqueta])
    axs[i].axis('off')
plt.show()

# Mostrar un ejemplo de cada clase usando datos['train'] y metadatos
fig, axs = plt.subplots(1, len(metadatos['clases']), figsize=(20, 3))
for idx, clase in enumerate(metadatos['clases']):
    for img, et in datos['train']:
        if et == idx:
            img = img.reshape(metadatos['tamano_img'], metadatos['tamano_img'])
            axs[idx].imshow(img, cmap='gray')
            axs[idx].set_title(clase)
            axs[idx].axis('off')
            break
plt.tight_layout()
plt.show()

#Manipular y visualizar el set
#Lo pasamos a TAMANO_IMG (100x100) y a blanco y negro (solo para visualizar)
plt.figure(figsize=(20,20))

TAMANO_IMG=100

for i, (imagen, etiqueta) in enumerate(datos['train'][:25]):
  imagen = imagen.reshape(TAMANO_IMG, TAMANO_IMG)
  plt.subplot(5, 5, i+1)
  plt.xticks([])
  plt.yticks([])
  plt.imshow(imagen, cmap='gray')
  plt.title(metadatos['clases'][etiqueta])
plt.show()

# Preparar arrays X (imágenes) y y (etiquetas) para entrenamiento
X = np.array([imagen for imagen, etiqueta in datos['train']])
y = np.array([etiqueta for imagen, etiqueta in datos['train']])

# Normalizar datos (0-255 → 0-1)
X = X.astype('float32') / 255.0

print(f"Forma de X: {X.shape}")
print(f"Forma de y: {y.shape}")
print(f"Número de clases: {len(metadatos['clases'])}")

print("\n=== CONFIGURACIÓN DE AUMENTO DE DATOS ===")

# Configurar generador de aumento de datos
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=15,
    zoom_range=[0.7, 1.4],
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

datagen.fit(X)

# Mostrar ejemplos de datos aumentados
plt.figure(figsize=(20, 8))
plt.suptitle('Ejemplos de Aumento de Datos', fontsize=16)

for imagen_aumentada, etiqueta in datagen.flow(X, y, batch_size=10, shuffle=False):
    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(imagen_aumentada[i].reshape(100, 100), cmap="gray")
        plt.title(f"'{metadatos['clases'][etiqueta[i]]}'")
    break
plt.tight_layout()
plt.show()

print("\n=== DIVISIÓN DE DATOS ===")

# Dividir datos en entrenamiento y validación (85% - 15%)
split_idx = int(len(X) * 0.85)
X_entrenamiento = X[:split_idx]
X_validacion = X[split_idx:]
y_entrenamiento = y[:split_idx]
y_validacion = y[split_idx:]

print(f"Datos de entrenamiento: {len(X_entrenamiento)} imágenes")
print(f"Datos de validación: {len(X_validacion)} imágenes")

print("\n=== CREACIÓN DEL MODELO CNN CON AUMENTO DE DATOS ===")

num_clases = len(metadatos['clases'])

# Crear modelo CNN optimizado para ASL con aumento de datos
modelo_asl = tf.keras.models.Sequential([
    # Primera capa convolucional
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(100, 100, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    
    # Segunda capa convolucional
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    
    # Tercera capa convolucional
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    
    # Regularización y capas densas
    tf.keras.layers.Dropout(0.5),  # Prevenir overfitting
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(num_clases, activation='softmax')  # Clasificación multiclase
])

# Compilar modelo
modelo_asl.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("Arquitectura del modelo:")
modelo_asl.summary()

print("\n=== CONFIGURACIÓN DE ENTRENAMIENTO ===")

# Configurar TensorBoard para monitoreo
tensorboard_callback = TensorBoard(
    log_dir='logs/asl_cnn_augmented',
    histogram_freq=1,
    write_graph=True,
    write_images=True
)

# Crear generador de datos de entrenamiento con aumento
data_gen_entrenamiento = datagen.flow(
    X_entrenamiento, 
    y_entrenamiento, 
    batch_size=32
)

# Configuración de entrenamiento
epochs = 100
batch_size = 32
steps_per_epoch = int(np.ceil(len(X_entrenamiento) / batch_size))
validation_steps = int(np.ceil(len(X_validacion) / batch_size))

print(f"Épocas: {epochs}")
print(f"Pasos por época: {steps_per_epoch}")

print("\n=== INICIANDO ENTRENAMIENTO ===")

# Entrenar el modelo
historial = modelo_asl.fit(
    data_gen_entrenamiento,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(X_validacion, y_validacion),
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    callbacks=[tensorboard_callback],
    verbose=1
)

print("\n=== ENTRENAMIENTO COMPLETADO ===")

# Guardar el modelo
nombre_modelo = 'modelo_asl_augmented.h5'
modelo_asl.save(nombre_modelo)
print(f"Modelo guardado como: {nombre_modelo}")

# Mostrar métricas finales
train_accuracy = historial.history['accuracy'][-1]
val_accuracy = historial.history['val_accuracy'][-1]
train_loss = historial.history['loss'][-1]
val_loss = historial.history['val_loss'][-1]

print(f"\n=== MÉTRICAS FINALES ===")
print(f"Precisión de entrenamiento: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
print(f"Precisión de validación: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
print(f"Pérdida de entrenamiento: {train_loss:.4f}")
print(f"Pérdida de validación: {val_loss:.4f}")

# Crear gráfica de entrenamiento
plt.figure(figsize=(15, 5))

# Gráfica de precisión
plt.subplot(1, 2, 1)
plt.plot(historial.history['accuracy'], label='Entrenamiento')
plt.plot(historial.history['val_accuracy'], label='Validación')
plt.title('Precisión del Modelo')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.legend()
plt.grid(True)

# Gráfica de pérdida
plt.subplot(1, 2, 2)
plt.plot(historial.history['loss'], label='Entrenamiento')
plt.plot(historial.history['val_loss'], label='Validación')
plt.title('Pérdida del Modelo')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

print(f"\n=== EXPORTACIÓN A TENSORFLOW.JS ===")

# Instalar tensorflowjs si no está instalado
try:
    import tensorflowjs as tfjs
    print("TensorFlow.js ya está instalado")
except ImportError:
    print("Instalando TensorFlow.js...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'tensorflowjs'])
    import tensorflowjs as tfjs

# Exportar modelo a TensorFlow.js
modelo_js_path = 'modelo_exportado/modelo_asl_js'
os.makedirs(modelo_js_path, exist_ok=True)

# Convertir modelo a formato TensorFlow.js
tfjs.converters.save_keras_model(modelo_asl, modelo_js_path)
print(f"Modelo exportado a TensorFlow.js en: {modelo_js_path}")

print(f"\n=== INFORMACIÓN DEL MODELO ENTRENADO ===")
print(f"Número de clases: {num_clases}")
print(f"Clases del modelo: {metadatos['clases']}")
print(f"\nArchivos generados:")
print(f"- Modelo Keras (.h5): {nombre_modelo}")
print(f"- Modelo TensorFlow.js: {modelo_js_path}/")
print(f"- Logs de TensorBoard: logs/asl_cnn_augmented/")

print(f"\nTipo de salida del modelo:")
print(f"- Array de probabilidades de forma (1, {num_clases})")
print(f"- Cada elemento representa la probabilidad de cada clase")
print(f"- La suma de todas las probabilidades es 1.0")
print(f"- Para obtener la predicción: np.argmax(modelo.predict(imagen))")

print("\n¡Modelo listo para usar en aplicaciones web con TensorFlow.js!")