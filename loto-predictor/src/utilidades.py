import os
import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime

def configurar_gpu(config_gpu):
    """Configura el uso de GPU según la configuración"""
    if config_gpu['habilitada']:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"✅ GPUs detectadas: {len(gpus)}")
                print(f"⚙️ Configuración: {config_gpu}")
            except RuntimeError as e:
                print(f"⚠️ Error configurando GPU: {e}")
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        print("🖥️ Usando solo CPU")

def guardar_resultados(combinaciones, ruta_resultados, nombre_archivo):
    """Guarda las combinaciones en un archivo CSV"""
    os.makedirs(ruta_resultados, exist_ok=True)
    fecha = datetime.now().strftime("%Y%m%d_%H%M%S")
    nombre_archivo = nombre_archivo.replace("{fecha}", fecha)
    ruta_completa = os.path.join(ruta_resultados, nombre_archivo)
    
    resultados_df = pd.DataFrame(
        combinaciones, 
        columns=[f'N{i+1}' for i in range(6)]
    )
    resultados_df.to_csv(ruta_completa, index=False)
    
    print(f"💾 Combinaciones guardadas en: {ruta_completa}")
    return ruta_completa