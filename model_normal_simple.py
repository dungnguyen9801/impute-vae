import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import copy
import utils
import encoders

class model_normal_simple():
    def __init__(self):
        self.encoder, self.decoder = self.get_model_gaussian_simple()

    def get_model_gaussian_simple(self):
        encoder = encoders.get_normal_simple_encoder()
        decoder = encoders.get_normal_simple_encoder()
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

