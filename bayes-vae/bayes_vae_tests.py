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

tf.keras.backend.set_floatx('float32')

def load_frey(shape=None, options=None):
    if shape:
        raise ValueError('shape should be None in load_frey')
    img_rows, img_cols = 28, 20
    ff = loadmat('../../data/freyface/frey_rawface.mat', squeeze_me=True, struct_as_record=False)
    return ff["ff"].T.reshape((-1, img_rows, img_cols))/255

def load_normal(shape=None, options=None):
    if shape is None:
        shape=(1000,1)
    return np.random.normal(0, options.get('scale', 1), shape)

test_cases = \
{
    'frey1':
    {
        'epochs': 2000000,
        'batch_size': 10,
        'input_shape': None,
        'model_dims': (256,2,256),
        'dataset_loader': load_frey,
        'model_class': mvb.model_vae_bayes,
        'options':
        {
            'length': 100,
            'seed': 0
        },
        'use_sgd':True,
        'report_frequency':1000
    },
    'normal1':
    {
        'epochs': 10000,
        'batch_size': 32,
        'input_shape': (4000,1),
        'dataset_loader': load_normal,
        'model_class': mns.model_normal_simple,
        'options':
        {
            'length': 100,
            'seed': 0,
            'scale': np.sqrt(2.0)
        },
        'use_sgd':False,
        'report_frequency': 100
    }
}

def run_test(test_case_name):
    test_case = test_cases[test_case_name]
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
    report_frequency=test_case.get('report_frequency',1)
    if not test_case.get('use_sgd'):
        batch_size = len(x)
    for epoch in range(epochs):
        loss_func = ec.elbo_calculator().get_loss_func()
        loss = train.train_one_epoch(model, x, optimizer, loss_func, y=None, batch_size=batch_size, options=options)
        if ((epoch + 1) % report_frequency == 0):
            print('epoch %s: loss = %s' %(epoch+1, -loss.numpy()))
            model.encoder.save('%s_encoder.h5' % test_case_name)
            model.decoder.save('%s_decoder.h5' % test_case_name)
    return model
