import sys
import time
sys.path.append('../shared')
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import copy
import model_normal_simple as mns
import model_vae_bayes as mvb
import elbo_calculator as ec
import train
import argparse
import os
from scipy.io import loadmat
import bayes_vae_tests as tests
test_case_name='frey1'
test_case = tests.test_cases[test_case_name]
epochs = test_case['epochs']
batch_size = test_case['batch_size']
input_shape = test_case['input_shape']
x = test_case['dataset_loader'](input_shape, test_case.get('options'))
if test_case['model_class'] == mns.model_normal_simple:
    model = mns.model_normal_simple()
else:
    dim_z_hidden, dim_z, dim_x_hidden = test_case['model_dims']
    model = test_case['model_class'](
        dim_z_hidden=dim_z_hidden,
        dim_z=dim_z,
        input_shape=x.shape[1:],
        dim_x_hidden=dim_x_hidden)
optimizer = tf.keras.optimizers.Adamax()
options = test_case.get('options')
print(options)
report_frequency=test_case.get('report_frequency',1)
start_time = time.time()
for epoch in range(epochs):
    loss_func = ec.elbo_calculator().get_loss_func_2()
    loss = train.train_one_random_batch(model, x, optimizer, loss_func, y=None, batch_size=batch_size, options=options)
    if ((epoch + 1) % report_frequency == 0):
        elapsed = time.time()-start_time
        print('epoch %s: loss = %s, elapsed = %s' %(epoch+1, -loss.numpy(), elapsed))
        if ((epoch + 1) % (report_frequency*10) == 0):
            model.encoder.save('%s_encoder.h5' % test_case_name)
            model.decoder.save('%s_decoder.h5' % test_case_name)
            z = np.random.normal(0,1,(1,dim_z))
            mu_x, log_sigma_x = model.decoder(z)
            img = np.reshape(mu_x.numpy(), (28,20))
            plt.imshow(img)
            plt.show()
        start_time = time.time()
