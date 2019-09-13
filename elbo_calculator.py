import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

class elbo_calculator():
    def __init__(self, vae_model, data):
        self.model = vae_model
        self.data = data
        
    # options = { 'length' : L, 'seed': seed }
    def get_elbo(self, options):
        model = self.model
        x = self.data
        zs = model.get_z_generator()(x, options)
        return -(tf.math.reduce_sum(model.get_func_log_p_z()(zs)) 
                + tf.math.reduce_sum(model.get_func_log_p_x_z()(zs, x))
                - tf.math.reduce_sum(model.get_func_log_q_z_x()(zs, x)))/len(zs)
