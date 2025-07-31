import numpy as np
from geneticalgorithm import geneticalgorithm as ga

def crear_funcion_aptitud(modelo, matriz_cuantica):
    """Crea función de aptitud para el algoritmo genético"""
    def funcion_aptitud(combinacion):
        # Penalizar combinaciones inválidas
        if len(np.unique(combinacion)) != 6:
            return 10000
        
        # Crear vector de la combinación
        vector_combo = np.zeros(49)
        for num in combinacion:
            if 1 <= num <= 49:
                vector_combo[num-1] = 1
        
        # Predecir con el modelo
        prob = modelo.predict(np.expand_dims(matriz_cuantica, 0), verbose=0)[0]
        puntaje = -np.sum(prob * vector_combo)  # Minimizar el negativo
        
        return puntaje
    
    return funcion_aptitud

def optimizar_combinaciones(modelo, matriz_cuantica, num_combinaciones=5, poblacion=100, iteraciones=200):
    """Ejecuta la optimización evolutiva para encontrar combinaciones óptimas"""
    funcion_aptitud = crear_funcion_aptitud(modelo, matriz_cuantica)
    
    # Configurar algoritmo genético
    parametros_ga = {
        'max_num_iteration': iteraciones,
        'population_size': poblacion,
        'mutation_probability': 0.15,
        'elit_ratio': 0.1,
        'crossover_probability': 0.7,
        'parents_portion': 0.3,
        'crossover_type': 'uniform',
        'max_iteration_without_improv': 50
    }
    
    optimizador = ga(
        function=funcion_aptitud,
        dimension=6,
        variable_type='int',
        variable_boundaries=np.array([[1, 49]] * 6),
        algorithm_parameters=parametros_ga
    )
    
    # Ejecutar optimización
    print("\n🧬 Iniciando optimización evolutiva...")
    resultado = optimizador.run()
    
    # Procesar mejores combinaciones
    mejor_combinacion = resultado['variable']
    return [sorted(mejor_combinacion)]