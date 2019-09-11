import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
tf.keras.backend.set_floatx('float32')

class elbo_calculator():
    def get_elbo(vae_model, data):
        
def get_elbo(x, z_generators, func_log_p_z_theta, func_log_q_z_x, func_log_p_x_z, options):
    zs = model.get_z_gaussian_generator()(x, options)
    return -(tf.math.reduce_sum(model.get_func_log_p_z_theta()(zs)) 
            + tf.math.reduce_sum(model.get_func_log_p_x_z()(zs, x))
            - tf.math.reduce_sum(model.get_func_log_q_z_x()(zs, x)))/len(zs)
