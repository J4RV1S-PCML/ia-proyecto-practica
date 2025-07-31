import numpy as np
import pandas as pd

def cargar_datos(ruta_archivo):
    """Carga datos históricos desde un archivo CSV"""
    return pd.read_csv(ruta_archivo)

def transformacion_cuantica(datos, num_numeros=49):
    """
    Transforma datos históricos en matriz de densidad cuántica
    """
    matriz_densidad = np.zeros((num_numeros, num_numeros))
    
    for _, row in datos.iterrows():
        estado = np.zeros(num_numeros)
        for i in range(1, 7):  # Columnas n1 a n6
            num = int(row[f'n{i}'])
            if 1 <= num <= num_numeros:
                estado[num-1] = 1
        
        norma = np.linalg.norm(estado)
        if norma > 0:
            estado /= norma
        
        matriz_densidad += np.outer(estado, estado)
    
    return matriz_densidad / len(datos)

def preparar_entrenamiento(matriz_cuantica, datos):
    """Prepara datos de entrenamiento para el modelo"""
    # Último sorteo como referencia
    ultimo_sorteo = datos.iloc[-1, 1:7].values.astype(int) - 1
    y_train = np.eye(49)[ultimo_sorteo].sum(axis=0)
    
    # Expandir dimensión para el modelo
    X_train = np.expand_dims(matriz_cuantica, axis=0)
    
    return X_train, y_train