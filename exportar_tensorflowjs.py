#!/usr/bin/env python3
"""
Script para exportar el modelo entrenado a TensorFlow.js
Ejecutar después de entrenar el modelo con index.py
"""

import os
import subprocess
import sys

def instalar_tensorflowjs():
    """Instalar TensorFlow.js si no está disponible"""
    try:
        import tensorflowjs
        print("✓ TensorFlow.js ya está instalado")
        return True
    except ImportError:
        print("📦 Instalando TensorFlow.js...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'tensorflowjs'])
            print("✓ TensorFlow.js instalado correctamente")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Error al instalar TensorFlow.js: {e}")
            return False

def exportar_modelo():
    """Exportar modelo de Keras a TensorFlow.js"""
    try:
        import tensorflow as tf
        import tensorflowjs as tfjs
        
        # Rutas de archivos
        modelo_keras = 'modelo_asl_augmented.h5'
        modelo_js_path = 'modelo_exportado/modelo_asl_js'
        
        # Verificar que el modelo existe
        if not os.path.exists(modelo_keras):
            print(f"❌ No se encontró el modelo: {modelo_keras}")
            print("Ejecuta primero el script index.py para entrenar el modelo")
            return False
        
        # Crear directorio de salida
        os.makedirs(modelo_js_path, exist_ok=True)
        
        # Cargar y convertir modelo
        print(f"📂 Cargando modelo desde: {modelo_keras}")
        modelo = tf.keras.models.load_model(modelo_keras)
        
        print(f"🔄 Convirtiendo modelo a TensorFlow.js...")
        tfjs.converters.save_keras_model(modelo, modelo_js_path)
        
        print(f"✅ Modelo exportado exitosamente a: {modelo_js_path}")
        
        # Listar archivos generados
        archivos = os.listdir(modelo_js_path)
        print(f"\n📁 Archivos generados:")
        for archivo in archivos:
            print(f"   - {archivo}")
        
        # Información adicional
        print(f"\n💡 Para usar en aplicaciones web:")
        print(f"   1. Copia la carpeta '{modelo_js_path}' a tu proyecto web")
        print(f"   2. Carga el modelo con: tf.loadLayersModel('ruta/model.json')")
        print(f"   3. Haz predicciones con: modelo.predict(tensor_imagen)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error durante la exportación: {e}")
        return False

def main():
    """Función principal"""
    print("🚀 Iniciando exportación del modelo a TensorFlow.js")
    print("=" * 50)
    
    # Paso 1: Instalar TensorFlow.js
    if not instalar_tensorflowjs():
        return
    
    # Paso 2: Exportar modelo
    if not exportar_modelo():
        return
    
    print("\n🎉 ¡Exportación completada exitosamente!")
    print("El modelo ahora puede usarse en aplicaciones web con TensorFlow.js")

if __name__ == "__main__":
    main()
