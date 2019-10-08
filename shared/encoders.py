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

def get_hi_vae_encoder(dim_input, dim_hidden, dim_latent, s_dim):
    eps = 0.0001
    x = keras.layers.Input(shape=(dim_input,))
    mu_x = tf.reduce_mean(x, axis=0)
    sigma_x = tf.sqrt(tf.reduce_mean((x-mu_x)**2, axis=0) + eps)
    x_norm = (x_mu_x)/sigma_x
    hidden_layer = keras.layers.Dense(dim_hidden)
    s_prop_layer = keras.layers.Dense(s_dim)
    x_s = hidden_layer(x_norm)
    s_prop = s_prop_layer(x_s)
    x_s = tf.keras.backend.repeat(x_s,s_dim)
    id_mat = tf.stack([tf.eye(s_dim)], axis=0)
    id_mat_batch = tf.tile(id_mat, tf.stack([tf.shape(x_s)[0], 1, 1]))
    x_s = tf.concat([x_s, id_mat_batch], axis=-1)
    mu_layer = keras.layers.Dense(dim_hidden, activation='linear')
    log_sigma_layer = keras.layers.Dense(dim_hidden, activation='linear')
    mus  = mu(x_s)
    log_sigmas = log_sigma_s(x_s)
    beta = tf.Variable(0.5)
    gamma = tf.Variable(0.5)
    return keras.models.Model(
        inputs=x,
        outputs= (s_prop, mus, log_sigmas, beta*1.0, gamma*1.0)
    )

def get_hi_vae_decoder(input_dim, latent_dim, s_dim, column_types):
    z = keras.layers.Input(shape=(latent_dim,))
    s = keras.layers.Input(shape=(s_dim,))
    beta = keras.layers.Input(shape=(1,))
    gamma = keras.layers.Input(shape=(1,))
    y = keras.layers.Dense(input_dim)
    y_s = tf.concat([y,s], axis=-1)

class hi_vae_mixed_gaussian_layer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(hi_vae_mixed_gaussian_layer, self).__init__(**kwargs)

    def build(self, dim_latent, mix_num):
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
        super(model_normal_simple_layer, self).build(imput_shape = (dim_latent,))  # Be sure to call this at the end

    def call(self, x):
        mu=x*self.a + self.b
        return mu, x * 0 + self.sigma

    def compute_output_shape(self, input_shape):
        return input_shape

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



        
