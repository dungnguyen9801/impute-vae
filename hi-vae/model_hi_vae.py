import sys
sys.path.append('../')
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from shared import utils, encoders

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
        decoder = encoders.get_hi_vae_decoder(latent_dim, s_dim, column_types)
        encoder = encoders.get_hi_vae_encoder(input_dim, hidden_dim, latent_dim, s_dim)
        return encoder, decoder

    def get_z_generator(self):
        def func(x, options=None):
            s_probs, mu_z, log_sigma_z, beta, gamma = self.encoder(x)
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
            eps = np.random.normal(0,1, size = (L, s_dim, batch, z_dim))
            return (tf.transpose(eps*sigma_z + mu_z, [1,0,2,3]), s_probs, beta, gamma)
        return func

    def get_func_log_p_xz(self):
        def func(zs, x):
            zs, s_probs, beta, gamma = zs
            L = tf.shape(zs)[1].numpy()
            s_dim = tf.shape(s_probs)[-1]
            batch = len(s_probs)
            output = self.decoder([
                tf.reshape(zs,(-1, zs.shape[-1])),
                beta,
                gamma])
            p_z = utils.get_gaussian_densities(
                tf.reshape(zs, [s_dim, batch, -1]),0,1)
            p_z = tf.math.reduce_sum(p_z, axis=-1) 
            p_x_z = 0
            i = 0
            for j, t in enumerate(self.column_types):
                if t == 1 or t == -1:
                    mu_x, log_sigma_x = output[j]
                    sigma_x = tf.math.exp(log_sigma_x)
                    p = utils.get_gaussian_densities(
                        x[:,i:i+1],
                        tf.reshape(mu_x, [-1, batch, 1]),
                        tf.reshape(sigma_x, [-1, batch, 1]))/L
                    p = tf.reshape(p, [s_dim, batch, -1])
                    p = tf.math.reduce_sum(p, axis=-1)
                    p_x_z = p_x_z + p
                    i += 1
                elif t == 0:
                    log_lambda_x = output[j][0]
                    log_lambda_x = tf.reshape(log_lambda_x, [-1,tf.shape(x)[0],1])
                    p = tf.nn.log_poisson_loss(
                            tf.broadcast_to(
                                x[:, i:i+1],
                                tf.shape(log_lambda_x)),
                            log_lambda_x)/L
                    p = tf.reshape(p, [s_dim, batch, -1])
                    p = tf.math.reduce_sum(p, axis=-1)
                    p_x_z = p_x_z + p
                    i += 1
                else:
                    probs = output[j][0]
                    probs = tf.reshape(probs,[-1, batch, t])
                    p = tf.math.log(tf.math.reduce_sum(probs * x[:,i:i+t], axis=-1))/L
                    p = tf.reshape(p, [s_dim, batch, -1])
                    p = tf.math.reduce_sum(p, axis=-1)
                    p_x_z = p_x_z + p
                    i += t
            return tf.math.reduce_sum(tf.transpose(s_probs) * (p_x_z + p_z))
        return func

    def get_func_log_q_z_x(self):
        def func(zs, x):
            zs, s_probs, _, _ = zs
            L = tf.shape(zs)[0].numpy()
            batch = x.shape[0]
            s_dim = tf.shape(s_probs)[-1]
            zs = tf.transpose(zs, [1,0,2,3])
            _, mu_z, log_sigma_z, _, _ = self.encoder(x)
            sigma_z = tf.math.exp(log_sigma_z)
            densities = tf.transpose(
                utils.get_gaussian_densities(zs, mu_z, sigma_z),
                [1,0,2,3])/L
            densities = tf.reshape(densities, [s_dim, batch, -1])
            densities = tf.math.reduce_sum(densities, axis=-1)
            return tf.math.reduce_sum(tf.transpose(s_probs) * densities)
        return func

    def get_trainable_variables(self):
        return [self.encoder.trainable_variables, self.decoder.trainable_variables]

