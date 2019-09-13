import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import utils

class model_vae_bayes():
    def __init__(self, dim_z_hidden, dim_z, input_shape, dim_x_hidden):
        self.encoder, self.decoder = self.get_model_vae_gaussian(
            dim_z_hidden, dim_z, input_shape, dim_x_hidden)
            
    def get_model_vae_gaussian(self, dim_z_hidden, dim_z, input_shape, dim_x_hidden):
        flatten_encode = keras.layers.Flatten()
        dense_encode = keras.layers.Dense(dim_z_hidden, activation='tanh')
        mu_encode = keras.layers.Dense(dim_z, activation='linear')
        log_sigma_encode = keras.layers.Dense(dim_z, activation='linear')
        inputs_encode = keras.layers.Input(shape=(*input_shape,))
        encoder=keras.models.Model(
            inputs=inputs_encode,
            outputs=(
                mu_encode(dense_encode(flatten_encode(inputs_encode))),
                log_sigma_encode(dense_encode(flatten_encode(inputs_encode)))
            )
        )
        
        dim_x = np.prod(input_shape)    
        dense_decode = keras.layers.Dense(dim_x_hidden, activation='tanh')
        mu_decode = keras.layers.Dense(dim_x, activation='linear')
        log_sigma_decode = keras.layers.Dense(dim_x, activation='linear')
        inputs_decode = keras.layers.Input(shape=(dim_z,))
        decoder=keras.models.Model(
            inputs=inputs_decode,
            outputs=(
                mu_decode(dense_decode((inputs_decode))),
                log_sigma_decode(dense_decode((inputs_decode)))
            )
        )
        
        return encoder, decoder

    def get_z_generator(self):
        def func(x, options):
            mu_z, log_sigma_z = self.encoder(x)
            sigma_z = tf.math.exp(log_sigma_z)
            if not options:
                L = 100
                seed = 0
            else:
                L = options['length']
                seed = options['seed']
            if seed:
                np.random.seed(seed)
            eps = np.random.normal(0,1, size = (L, x.shape[0], mu_z.shape[1]))
            return eps* sigma_z + mu_z
        return func

    def get_func_log_p_z(self):
        def func(zs):
            return -zs**2/2 - 0.5 * tf.math.log(2*np.pi)
        return func

    def get_func_log_q_z_x(self):
        def func(zs, x):
            mu_z, log_sigma_z = self.encoder(x)
            sigma_z = tf.math.exp(log_sigma_z)
            return utils.get_gaussian_densities(zs, mu_z, sigma_z)
        return func

    def get_func_log_p_x_z(self):
        def func(zs, x):
            mu_x, log_sigma_x = self.decoder(tf.reshape(zs,(-1, zs.shape[2])))
            sigma_x = tf.math.exp(log_sigma_x)
            mu_x = tf.reshape(mu_x, (-1, *x.shape))
            sigma_x = tf.reshape(sigma_x,(-1, *x.shape))
            return utils.get_gaussian_densities(x, mu_x, sigma_x)
        return func

    def get_trainable_variables(self):
        return [self.encoder.trainable_variables, self.decoder.trainable_variables]

