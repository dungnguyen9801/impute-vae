import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import copy
import utils

class normal_model():
    def __init__(self):
        self.encoder, self.decoder = self.get_model_gaussian_simple()

    def get_model_gaussian_simple(self):
        mu_encode = keras.layers.Dense(1, activation='linear')
        sigma_encode = tf.Variable(1.0)
                
        inputs_encode = keras.layers.Input(shape=(1,))
        encoder=keras.models.Model(
            inputs=inputs_encode,
            outputs=(
                mu_encode(inputs_encode),
                sigma_encode
            )
        )

        mu_encode = keras.layers.Dense(1, activation='linear')
        sigma_encode = tf.Variable(1.0)    
        inputs_decode = keras.layers.Input(shape=(1,))
        decoder=keras.models.Model(
            inputs=inputs_decode,
            outputs=(
                mu_decode(inputs_decode),
                sigma_decode
            )
        )
        
        return encoder, decoder

    def get_z_gaussian_generator(self):
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

    def get_func_log_p_z_gaussian(self):
        def func(zs):
            return -zs**2/2 - 0.5 * tf.math.log(2*np.pi)
        return func

    def get_func_log_q_z_x_gaussian(self):
        def func(zs, x):
            mu_z, sigma_z = self.encoder(x)
            return utils.get_gaussian_densities(zs, mu_z, sigma_z)
        return func

    def get_func_log_p_x_z_gaussian(self.decoder):
        def func(zs, x):
            mu_x, sigma_x = decoder(tf.reshape(zs,(-1, 1)))
            mu_x = tf.reshape(mu_x, (-1, *x.shape))
            return get_gaussian_densities(x, mu_x, sigma_x)
        return func
