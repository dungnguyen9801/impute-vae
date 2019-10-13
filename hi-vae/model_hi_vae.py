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
            input_dim, 
            hidden_dim,
            latent_dim,
            s_dim,
            column_types)
        self.column_types = column_types
            
    def get_hi_vae_encoder(
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
            s_dim, batch, z_dim = mu_z.shape
            eps = np.random.normal(0,1, size = (s_dim, batch, L, z_dim))
            return s_prop, beta, gamma, eps* tf.reshape(
                sigma_z, (s_dim, batch, 1, z_dim)) + tf.reshape(mu_z,
                (s_dim, batch, 1, z_dim))
        return func

    def get_func_log_p_xz(self):
        def func(zs, x):
            zs, s_prop, beta, gamma = zs
            s_dim = len(s_prop)
            output = self.decoder([
                tf.reshape(zs,(-1, zs.shape[-1])),
                s_prop,
                beta,
                gamma])
            p_z = utils.get_gaussian_densities(
                tf.reshape(zs, (s_dim, -1)),0,1)
            p_z = tf.math.reduce_sum(p_z, axis=1) 
            p_x_z = 0
            i = 0
            for j, t in enumerate(self.column_types):
                if t == 1:
                    mu_x, log_sigma_x = output[:,j]
                    sigma_x = tf.math.exp(log_sigma_x)
                    p = utils.get_gaussian_densities(
                        x[:,i], mu_x, sigma_x)
                    p = tf.reshape(p, (s_dim, -1))
                    p_x_z = p_x_z + p
                    i += 1
                elif t == 0:
                    log_lambda_x = output[:, j]
                    p = tf.nn.log_poisson_loss(
                            x[:, i],
                            log_lambda_x)
                    p = tf.reshape(p, (s_dim, -1))
                    p_x_z = p_x_z + p
                    i += 1
                else:
                    p = tf.math.log(output[:,j] * x[:,i:i+t])
                    p = tf.reshape(p, (s_dim, -1))
                    p_x_z = p_x_z + p
                    i += t
            return tf.matmul(s_prop, p_x_z + p_z)
        return func

    def get_func_log_q_z_x(self):
        def func(zs, x):
            mu_z, log_sigma_z = self.encoder(x)
            sigma_z = tf.math.exp(log_sigma_z)
            return tf.math.reduce_sum(utils.get_gaussian_densities(zs, mu_z, sigma_z))
        return func

    def get_trainable_variables(self):
        return [self.encoder.trainable_variables, self.decoder.trainable_variables]

