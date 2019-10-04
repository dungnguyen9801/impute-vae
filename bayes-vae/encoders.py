import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import utils

def get_bayes_encoder(input_shape, dim_hidden, dim_latent, activation_mu='linear', activation_sigma='linear'):
    flatten_encode = keras.layers.Flatten()
    dense_encode = keras.layers.Dense(dim_hidden, activation='tanh')
    mu_encode = keras.layers.Dense(dim_latent, activation=activation_mu)
    log_sigma_encode = keras.layers.Dense(dim_latent, activation=activation_sigma)
    inputs_encode = keras.layers.Input(shape=(*input_shape,))
    return keras.models.Model(
        inputs=inputs_encode,
        outputs=(
            mu_encode(dense_encode(flatten_encode(inputs_encode))),
            log_sigma_encode(dense_encode(flatten_encode(inputs_encode)))
        )
    )

class model_normal_simple_layer(keras.layers.Layer):

    def __init__(self, **kwargs):
        super(model_normal_simple_layer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        simple_init = keras.initializers.Constant(value=0.5)
        self.a = self.add_weight(
            name='a',
            shape=None,
            initializer=simple_init,
            dtype='float32',
            trainable=True)
        self.b = self.add_weight(
            name='b',
            shape=None,
            initializer=simple_init,
            dtype='float32',
            trainable=True)
        self.sigma = self.add_weight(
            name='sigma',
            shape=None,
            initializer=simple_init,
            dtype='float32',
            trainable=True)
        super(model_normal_simple_layer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        mu=x*self.a + self.b
        return mu, x * 0 + self.sigma

    def compute_output_shape(self, input_shape):
        return input_shape

def get_normal_simple_encoder():
    inputs_encode = keras.layers.Input(shape=(1,))
    layer_encode = model_normal_simple_layer()
    return keras.models.Model(
        inputs=inputs_encode,
        outputs=layer_encode(inputs_encode)
    )



        