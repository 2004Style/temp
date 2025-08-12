# Ejemplo de cómo usar el modelo entrenado para hacer predicciones
import numpy as np
import cv2
import tensorflow as tf

# Cargar el modelo entrenado
modelo = tf.keras.models.load_model('modelo-cnn-ad.h5')

# Clases que reconoce el modelo (en el mismo orden que durante el entrenamiento)
clases = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
          'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
          'del', 'nothing', 'space']

def predecir_gesto(ruta_imagen):
    """
    Función para predecir el gesto de una imagen
    
    Args:
        ruta_imagen (str): Ruta a la imagen a clasificar
    
    Returns:
        tuple: (clase_predicha, probabilidad_maxima, todas_las_probabilidades)
    """
    # Cargar y preprocesar la imagen
    img = cv2.imread(ruta_imagen)
    if img is None:
        return None, None, None
    
    # Preprocesar igual que en el entrenamiento
    img = cv2.resize(img, (100, 100))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.reshape(100, 100, 1)
    img = img.astype(float) / 255.0  # Normalizar
    img = np.expand_dims(img, axis=0)  # Agregar dimensión de batch
    
    # Hacer predicción
    prediccion = modelo.predict(img)
    
    # Obtener la clase con mayor probabilidad
    indice_clase = np.argmax(prediccion[0])
    clase_predicha = clases[indice_clase]
    probabilidad_maxima = prediccion[0][indice_clase]
    
    return clase_predicha, probabilidad_maxima, prediccion[0]

def mostrar_top_predicciones(ruta_imagen, top_n=5):
    """
    Muestra las top N predicciones con sus probabilidades
    """
    clase, prob, todas_prob = predecir_gesto(ruta_imagen)
    
    if clase is None:
        print("Error: No se pudo cargar la imagen")
        return
    
    # Obtener índices de las probabilidades más altas
    indices_ordenados = np.argsort(todas_prob)[::-1]
    
    print(f"\nPredicciones para la imagen: {ruta_imagen}")
    print("-" * 40)
    for i in range(min(top_n, len(clases))):
        idx = indices_ordenados[i]
        print(f"{i+1}. {clases[idx]}: {todas_prob[idx]:.4f} ({todas_prob[idx]*100:.2f}%)")

# Ejemplo de uso
if __name__ == "__main__":
    # Ejemplo de predicción
    # ruta_test = "ruta/a/tu/imagen/test.jpg"
    # clase, prob, _ = predecir_gesto(ruta_test)
    # print(f"Predicción: {clase} (Probabilidad: {prob:.4f})")
    # mostrar_top_predicciones(ruta_test)
    
    print("=== INFORMACIÓN DEL MODELO ===")
    print(f"El modelo reconoce {len(clases)} clases:")
    for i, clase in enumerate(clases):
        print(f"{i}: {clase}")
    
    print("\n=== FORMATO DE SALIDA ===")
    print("modelo.predict(imagen) devuelve:")
    print("- Array de forma (1, 29) con probabilidades")
    print("- Cada posición corresponde a una clase específica")
    print("- La suma de todas las probabilidades es 1.0")
    print("- Para obtener la clase: np.argmax(prediccion)")
