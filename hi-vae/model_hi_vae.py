def get_model_hi_vae():
    batch_norm = tf.keras.layers.BatchNormalization()
    


import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import utils
import encoders

class model_hi_vae():
    def __init__(self, dim_z_hidden, dim_z, input_shape, dim_x_hidden):
        self.encoder, self.decoder = self.get_model_vae_gaussian(
            dim_z_hidden, dim_z, input_shape, dim_x_hidden)
            
    def get_model_vae_gaussian(self, dim_z_hidden, dim_z, input_shape, dim_x_hidden):
        encoder = encoders.get_bayes_encoder(input_shape, dim_z_hidden, dim_z)
        decoder = encoders.get_bayes_encoder((dim_z,), dim_x_hidden, np.prod(input_shape), activation_mu='sigmoid')
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
            mu_x, log_sigma_x = self.decoder(tf.reshape(zs,(-1, zs.shape[-1])))
            sigma_x = tf.math.exp(log_sigma_x)
            mu_x = tf.reshape(mu_x, (-1, *x.shape))
            sigma_x = tf.reshape(sigma_x,(-1, *x.shape))
            return utils.get_gaussian_densities(x, mu_x, sigma_x)
        return func

    def get_trainable_variables(self):
        return [self.encoder.trainable_variables, self.decoder.trainable_variables]

