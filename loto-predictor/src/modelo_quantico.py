import tensorflow as tf
from tensorflow.keras.layers import Layer, Input, Conv1D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model

class QuantumAttention(Layer):
    """Capa de atención cuántica personalizada"""
    def __init__(self, **kwargs):
        super(QuantumAttention, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], input_shape[-1]),
            initializer='random_normal',
            trainable=True
        )
        super(QuantumAttention, self).build(input_shape)
    
    def call(self, inputs):
        # Simular medición cuántica: |ψ|^2
        prob = tf.math.square(tf.math.abs(inputs))
        
        # Mecanismo de atención
        att = tf.nn.softmax(tf.matmul(prob, self.w))
        return inputs * att

def crear_modelo_quantico(input_shape=(49, 49)):
    """Crea el modelo neuronal con atención cuántica"""
    inputs = Input(shape=input_shape)
    
    # Atención cuántica
    x = QuantumAttention()(inputs)
    
    # Capas convolucionales
    x = Conv1D(64, 3, activation='relu')(x)
    x = Flatten()(x)
    
    # Capas densas
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(49, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model