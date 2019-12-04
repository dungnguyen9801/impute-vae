import sys
sys.path.append('../')
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from shared import utils
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
        decoder = encoders.get_hi_vae_decoder(
            column_types,
            latent_dim,
            s_dim)
        encoder = encoders.get_hi_vae_encoder(
            column_types,
            input_dim, hidden_dim,
            latent_dim,
            s_dim)
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
            x_miss_list = x
            x = x_miss_list[: tf.shape(x_miss_list)[1]//2]
            zs, s_probs, beta, gamma = zs
            L = tf.shape(zs)[1].numpy()
            s_dim = tf.shape(s_probs)[-1]
            batch = len(s_probs)
            output, s_component_means = self.decoder([
                tf.reshape(zs,(-1, zs.shape[-1])),
                beta,
                gamma])
            p_z = utils.get_gaussian_densities(
                tf.reshape(zs, [-1]), 
                tf.reshape(tf.broadcast_to(
                    tf.reshape(s_component_means, [-1, 1]), [s_dim, np.prod(tf.shape(zs))]),
                1)
            p_z = tf.reshape(p_z, [s_dim, s_dim, -1])
            p_z = tf.math.reduce_sum(p_z, axis=[1,2])/s_dim 
            p_x_z = 0
            i = 0
            for j, column in enumerate(self.column_types):
                type_, dim = column['type'], column['dim']
                if type_ == 'real' or type_ == 'pos':
                    mu_x, log_sigma_x = output[j]
                    sigma_x = tf.math.exp(log_sigma_x)
                    p = utils.get_gaussian_densities(
                        x[:,i:i+dim],
                        tf.reshape(mu_x, [-1, batch, 1]),
                        tf.reshape(sigma_x, [-1, batch, 1]))
                    p = tf.reshape(p, [s_dim, batch, -1])
                    p = tf.math.reduce_sum(p, axis=-1)
                    p_x_z = p_x_z + p
                elif type_ == 'count':
                    log_lambda_x = output[j][0]
                    log_lambda_x = tf.reshape(log_lambda_x, [-1,tf.shape(x)[0],1])
                    p = tf.nn.log_poisson_loss(
                            tf.broadcast_to(
                                x[:, i:i+dim],
                                tf.shape(log_lambda_x)),
                            log_lambda_x)
                    p = tf.reshape(p, [s_dim, batch, -1])
                    p = tf.math.reduce_sum(p, axis=-1)
                    p_x_z = p_x_z + p
                else:
                    probs = output[j][0]
                    probs = tf.reshape(probs,[-1, batch, dim])
                    p = tf.math.log(tf.math.reduce_sum(probs * x[:,i:i+dim], axis=-1))
                    p = tf.reshape(p, [s_dim, batch, -1])
                    p = tf.math.reduce_sum(p, axis=-1)
                    p_x_z = p_x_z + p
                i += dim
            return tf.math.reduce_sum(tf.transpose(s_probs) * p_x_z)/L +
                     tf.math.reduce_sum(tf.transpose(s_probs) * p_z))/L
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
                [1,0,2,3])
            densities = tf.reshape(densities, [s_dim, batch, -1])
            densities = tf.math.reduce_sum(densities, axis=-1)
            return tf.math.reduce_sum(tf.transpose(s_probs) * densities)/L
        return func

    def get_trainable_variables(self):
        return [self.encoder.trainable_variables, self.decoder.trainable_variables]

