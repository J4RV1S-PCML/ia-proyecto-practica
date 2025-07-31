import numpy as np
import yaml
import os
import sys
from datetime import datetime
import tensorflow as tf

# Agregar ruta para importaciones
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocesamiento import cargar_datos, transformacion_cuantica, preparar_entrenamiento
from modelo_quantico import crear_modelo_quantico
from optimizacion_evolutiva import optimizar_combinaciones
from utilidades import configurar_gpu, guardar_resultados

def main():
    # Cargar configuración
    config_path = os.path.join(os.path.dirname(__file__), '../config/config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Configurar GPU
    configurar_gpu(config['gpu'])
    
    # Cargar datos históricos
    ruta_datos = os.path.join(os.path.dirname(__file__), '../', config['data']['ruta_historico'])
    datos = cargar_datos(ruta_datos)
    
    # Preprocesamiento cuántico
    matriz_cuantica = transformacion_cuantica(datos, config['modelo']['num_numeros'])
    
    # Preparar datos de entrenamiento
    X_train, y_train = preparar_entrenamiento(matriz_cuantica, datos)
    
    # Crear y entrenar modelo
    modelo = crear_modelo_quantico()
    print("\n🔮 Entrenando modelo cuántico-neural...")
    modelo.fit(
        X_train, 
        y_train.reshape(1, -1),
        epochs=config['entrenamiento']['epocas'],
        batch_size=1,
        verbose=1
    )
    
    # Optimización evolutiva
    combinaciones = optimizar_combinaciones(
        modelo,
        matriz_cuantica,
        num_combinaciones=config['optimizacion']['num_combinaciones'],
        poblacion=config['optimizacion']['poblacion'],
        iteraciones=config['optimizacion']['iteraciones']
    )
    
    # Guardar resultados
    ruta_resultados = os.path.join(os.path.dirname(__file__), '../', config['data']['ruta_resultados'])
    ruta_guardado = guardar_resultados(
        combinaciones,
        ruta_resultados,
        config['data']['nombre_archivo']
    )
    
    print("\n✅ Proceso completado! Combinaciones guardadas en:", ruta_guardado)

if __name__ == "__main__":
    main()