import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import copy
import utils


class model_normal_simple_layer(keras.layers.Layer):

    def __init__(self, **kwargs):
        super(model_normal_simple_layer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.a = self.add_weight(
            name='a',
            shape=None,
            initializer='uniform',
            dtype='float32',
            trainable=True)
        self.b = self.add_weight(
            name='b',
            shape=None,
            initializer='uniform',
            dtype='float32',
            trainable=True)
        self.sigma = self.add_weight(
            name='sigma',
            shape=None,
            initializer='uniform',
            dtype='float32',
            trainable=True)
        super(model_normal_simple_layer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        mu=x*self.a + self.b
        return mu, x * 0 + self.sigma

    def compute_output_shape(self, input_shape):
        return input_shape

class model_normal_simple():
    def __init__(self):
        self.encoder, self.decoder = self.get_model_gaussian_simple()

    def get_model_gaussian_simple(self):
        inputs_encode = keras.layers.Input(shape=(1,))
        layer_encode = model_normal_simple_layer()
        encoder=keras.models.Model(
            inputs=inputs_encode,
            outputs=layer_encode(inputs_encode)
        )

        inputs_decode = keras.layers.Input(shape=(1,))
        layer_decode = model_normal_simple_layer()
        decoder=keras.models.Model(
            inputs=inputs_decode,
            outputs=layer_decode(inputs_decode)
        )
        
        return encoder, decoder

    def get_z_generator(self):
        def func(x, options):
            mu_z, sigma_z = self.encoder(x)
            if not options:
                L = 100
                seed = 0
            else:
                L = options['length']
                seed = options['seed']
            if seed:
                np.random.seed(seed)
            eps = np.random.normal(0,1, size = (L, x.shape[0], mu_z.shape[1]))
            return eps * sigma_z + mu_z
        return func

    def get_func_log_p_z(self):
        def func(zs):
            return -zs**2/2 - 0.5 * tf.math.log(2*np.pi)
        return func

    def get_trainable_variables(self):
        return [self.encoder.trainable_variables, self.decoder.trainable_variables]

    def get_func_log_q_z_x(self):
        def func(zs, x):
            mu_z, sigma_z = self.encoder(x)
            return utils.get_gaussian_densities(zs, mu_z, sigma_z)
        return func

    def get_func_log_p_x_z(self):
        def func(zs, x):
            mu_x, sigma_x = self.decoder(tf.reshape(zs,(-1, 1)))
            mu_x = tf.reshape(mu_x, (-1, *x.shape))
            sigma_x = tf.reshape(sigma_x,(-1, *x.shape))
            return utils.get_gaussian_densities(x, mu_x, sigma_x)
        return func

    def get_trainable_variables(self):
        return [self.encoder.trainable_variables, self.decoder.trainable_variables]

