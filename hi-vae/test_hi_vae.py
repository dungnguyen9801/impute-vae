import sys
import time
sys.path.append('../shared')
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import copy
import model_hi_vae as mhv
import elbo_calculator as ec
import train
import os
from scipy.io import loadmat
import hi_vae_tests as tests
#%load_ext autoreload
test_case_name='hi_vae_wine'
test_case = tests.test_cases[test_case_name]
epochs = test_case['epochs']
batch_size = test_case['batch_size']
input_shape = test_case['input_shape']
x, column_types = test_case['dataset_loader']()
input_dim = x.shape[-1]
hidden_dim =3
latent_dim=2
s_dim=2
optimizer = tf.keras.optimizers.Adamax()
options = test_case.get('options')
report_frequency=test_case.get('report_frequency',1)
model = mhv.model_hi_vae(input_dim, hidden_dim, latent_dim, s_dim, column_types)
start_time = time.time()
for epoch in range(epochs):
    loss_func = ec.elbo_calculator().get_loss_func_2()
    loss = train.train_one_random_batch(model, x, optimizer, loss_func, y=None, batch_size=batch_size, options=options)
    if ((epoch + 1) % report_frequency == 0):
        elapsed = time.time()-start_time
        print('epoch %s: loss = %s, elapsed = %s' %(epoch+1, -loss.numpy(), elapsed))
        start_time = time.time()
