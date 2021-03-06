import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

class elbo_calculator():
    def __init__(self, vae_model=None, data=None):
        self.model = vae_model
        self.data = data
        
    # options = { 'length' : L, 'seed': seed }
    def get_loss_func_2(self):
        def func(model, x,y=None,options=None):
            zs = model.get_z_generator()(x, options)
            return -(model.get_func_log_p_xz()(zs, x)
                - model.get_func_log_q_z_x()(zs, x))/len(x)
        return func
