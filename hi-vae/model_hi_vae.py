import sys
sys.path.append('../shared')
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import utils
import encoders

class model_hi_vae():
    def __init__(self, input_dim, hidden_dim, latent_dim, s_dim, column_types):
        self.encoder, self.decoder = self.get_hi_vae_encoder(
            dim_z_hidden, dim_z, input_shape, dim_x_hidden)
            
    def get_model_hi_vae(
            self, 
            input_dim, 
            hidden_dim,
            latent_dim,
            s_dim,
            column_types):
        encoder = encoders.get_hi_vae_encoder(input_dim, hidden_dim, latent_dim, s_dim)
        decoder = encoders.get_hi_vae_decoder(latent_dim, s_dim, column_types)
        return encoder, decoder

    def get_z_generator(self):
        def func(x, options):
            s_prop, mu_z, log_sigma_z, beta, gamma = self.encoder(x)
            sigma_z = tf.math.exp(log_sigma_z)
            if not options:
                L = 100
                seed = 0
            else:
                L = options['length']
                seed = options['seed']
            if seed:
                np.random.seed(seed)
            eps = np.random.normal(0,1, size = (L, *mu_z.shape))
            return s_prop, beta, gamma, eps* sigma_z + mu_z
        return func

    def get_func_log_p_xz(self):
        def func(zs, x):
            zs, s_prop = zs
            mu_x, log_sigma_x = self.decoder(tf.reshape(zs,(-1, zs.shape[-1])))
            sigma_x = tf.math.exp(log_sigma_x)
            mu_x = tf.reshape(mu_x, (-1, *x.shape))
            sigma_x = tf.reshape(sigma_x,(-1, *x.shape))
            return tf.math.reduce_sum(utils.get_gaussian_densities(x, mu_x, sigma_x)) \
                + tf.math.reduce_sum(utils.get_gaussian_densities(zs,0,1))
        return func

    def get_func_log_q_z_x(self):
        def func(zs, x):
            mu_z, log_sigma_z = self.encoder(x)
            sigma_z = tf.math.exp(log_sigma_z)
            return tf.math.reduce_sum(utils.get_gaussian_densities(zs, mu_z, sigma_z))
        return func

    def get_trainable_variables(self):
        return [self.encoder.trainable_variables, self.decoder.trainable_variables]

