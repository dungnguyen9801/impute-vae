import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import utils

def get_bayes_encoder(input_shape, hidden_dim, latent_dim, activation_mu='linear', activation_sigma='linear'):
    flatten_encode = keras.layers.Flatten()
    dense_encode = keras.layers.Dense(hidden_dim, activation='tanh')
    mu_encode = keras.layers.Dense(latent_dim, activation=activation_mu)
    log_sigma_encode = keras.layers.Dense(latent_dim, activation=activation_sigma)
    inputs_encode = keras.layers.Input(shape=(*input_shape,))
    return keras.models.Model(
        inputs=inputs_encode,
        outputs=(
            mu_encode(dense_encode(flatten_encode(inputs_encode))),
            log_sigma_encode(dense_encode(flatten_encode(inputs_encode)))
        )
    )

def get_hi_vae_encoder(input_dim, hidden_dim, latent_dim, s_dim):
    eps = 0.0001
    x = keras.layers.Input(shape=(input_dim,))
    mu_x = tf.reduce_mean(x, axis=0)
    sigma_x = tf.sqrt(tf.reduce_mean((x-mu_x)**2, axis=0) + eps)
    x_norm = (x-mu_x)/sigma_x
    hidden_layer = keras.layers.Dense(hidden_dim, activation='tanh')
    s_probs_layer = keras.layers.Dense(s_dim, activation='softmax')
    x_s = hidden_layer(x_norm)
    s_probs = s_probs_layer(x_s)
    x_s = tf.keras.backend.repeat(x_s,s_dim)
    id_mat = tf.stack([tf.eye(s_dim)], axis=0)
    id_mat_batch = tf.tile(id_mat, tf.stack([tf.shape(x_s)[0], 1, 1], name='stack_x'))
    x_s = tf.concat([x_s, id_mat_batch], axis=-1)
    mu = keras.layers.Dense(latent_dim, activation='linear')
    log_sigma = keras.layers.Dense(latent_dim, activation='linear')
    mu_z  = mu(x_s)
    log_sigma_z = log_sigma(x_s)
    mu_z = tf.transpose(mu_z, [1,0,2])
    log_sigma_z = tf.transpose(log_sigma_z, [1,0,2])
    beta = tf.Variable(0.0)
    gamma = tf.Variable(1.0)
    return keras.models.Model(
        inputs=x,
        outputs=(s_probs, mu_z, log_sigma_z, beta+ .0*x[0,0], gamma + .0*x[0,0])
    )

def get_hi_vae_decoder(latent_dim, s_dim, column_types):
    z = keras.layers.Input(shape=(latent_dim,))
    beta = keras.layers.Input(shape=(1,))
    gamma = keras.layers.Input(shape=(1,))
    input_dim = len(column_types)
    shared_layer = keras.layers.Dense(input_dim)
    y = shared_layer(z)
    s_tail = tf.keras.backend.repeat(
        tf.eye(s_dim),
        tf.shape(z)[0]//s_dim)
    s_tail = tf.reshape(s_tail, (-1, s_dim))
    prop_layers = []
    for t in column_types:
        if t == 0:
            prop_layers.append((keras.layers.Dense(1),))
        elif t == 1 or t == -1:
            prop_layers.append((keras.layers.Dense(1), keras.layers.Dense(1)))
        else:
            prop_layers.append((keras.layers.Dense(t, activation='softmax'),))
    output = []
    for d in range(input_dim):
        y_d_s = tf.concat(
            [y[:,d:d+1], s_tail],
            axis=-1)
        output.append(list(map(lambda f: f(y_d_s), prop_layers[d])))
        if column_types[d] == 1:
            output[d][0] = output[d][0] * gamma + beta
            output[d][1] = output[d][1] * gamma
    return keras.models.Model(
        inputs=([z, beta, gamma]),
        outputs= output
    )

class hi_vae_mixed_gaussian_layer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(hi_vae_mixed_gaussian_layer, self).__init__(**kwargs)

    def build(self, latent_dim, mix_num):
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
        super(model_normal_simple_layer, self).build(imput_shape = (latent_dim,))  # Be sure to call this at the end

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



        
