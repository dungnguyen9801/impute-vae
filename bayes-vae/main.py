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

def test_elbo1():
    epochs=10000
    batch_size=32
    x = np.random.normal(0,1, size=(1200,3,2))
    model = mvb.model_vae_bayes(dim_z_hidden=4, dim_z=2, input_shape=x.shape[1:], dim_x_hidden=3)
    elbo_cal = ec.elbo_calculator(model, x)
    optimizer = tf.keras.optimizers.Adamax()
    options = {'length': 100, 'seed': 0}
    for epoch in range(epochs):
        loss = lambda : elbo_cal.get_elbo(options)
        optimizer.minimize(loss, model.get_trainable_variables())
        if (epoch+1) % 1000 == 0:
            print('epoch %s: loss = %s' %(epoch+1, loss().numpy()))

def test_elbo1a():
    epochs=10000
    batch_size=32
    x = np.random.normal(0,1, size=(1200,3,2))
    model = mvb.model_vae_bayes(dim_z_hidden=4, dim_z=2, input_shape=x.shape[1:], dim_x_hidden=3)
    optimizer = tf.keras.optimizers.Adamax()
    options = {'length': 100, 'seed': 0}
    for epoch in range(epochs):
        elbo_cal = ec.elbo_calculator()
        elbo_func = elbo_cal.get_loss_func()
        loss = lambda : elbo_func(model, x, y=None, options=options)
        optimizer.minimize(loss, model.get_trainable_variables())
        if (epoch+1) % 1 == 0:
            print('epoch %s: loss = %s' %(epoch+1, loss().numpy()))

def test_elbo1b():
    epochs=10000
    batch_size=32
    x = np.random.normal(0,1, size=(1200,3,2))
    model = mvb.model_vae_bayes(dim_z_hidden=4, dim_z=2, input_shape=x.shape[1:], dim_x_hidden=3)
    optimizer = tf.keras.optimizers.Adamax()
    options = {'length': 100, 'seed': 0}
    for epoch in range(epochs):
        loss_func = ec.elbo_calculator().get_loss_func()
        loss = train.train_one_epoch(model, x, optimizer, loss_func, y=None, batch_size=batch_size, options=options)
        if (epoch+1) % 1 == 0:
            print('epoch %s: loss = %s' %(epoch+1, loss.numpy()))


def test_elbo2():
    epochs=10000
    batch_size=32
    zs = np.random.normal(0,1, (1000,1))
    xs = zs + np.random.normal(0,1, (1000,1))
    model = mns.model_normal_simple()
    elbo_cal = ec.elbo_calculator(model, xs)
    optimizer = tf.keras.optimizers.Adamax()
    options = {'length': 100, 'seed': 0}
    for epoch in range(epochs):
        loss_elbo = lambda : elbo_cal.get_elbo(options)
        optimizer.minimize(loss_elbo, model.get_trainable_variables())
        if (epoch+1) % 200 == 0:
            print('epoch %s: loss = %s' %(epoch + 1, loss_elbo().numpy()))
            
def test_elbo2a():
    epochs=10000
    batch_size=1000
    zs = np.random.normal(0,1, (1000,1))
    xs = zs + np.random.normal(0,1, (1000,1))
    model = mns.model_normal_simple()
    optimizer = tf.keras.optimizers.Adamax()
    options = {'length': 100, 'seed': 0}
    for epoch in range(epochs):
        loss_func = ec.elbo_calculator().get_loss_func()
        loss = train.train_one_epoch(model, xs, optimizer, loss_func, y=None, batch_size=batch_size, options=options)
        if (epoch+1) % 100 == 0:
            print('epoch %s: loss = %s' %(epoch+1, loss.numpy()))
            
def test_elbo1():
    epochs=10000
    batch_size=32
    x = np.random.normal(0,1, size=(1200,1))
    model = mvb.model_vae_bayes(dim_z_hidden=1, dim_z=1, input_shape=x.shape[1:], dim_x_hidden=1)
    elbo_cal = ec.elbo_calculator()
    optimizer = tf.keras.optimizers.Adamax()
    options = {'length': 100, 'seed': 0}
    for epoch in range(epochs):
        loss_func = ec.elbo_calculator().get_loss_func()
        loss = train.train_one_epoch(model, x, optimizer, loss_func, y=None, batch_size=batch_size, options=options)
        if (epoch+1) % 100 == 0:
            print('epoch %s: loss = %s' %(epoch+1, -loss.numpy()))
            
def test_elbo4():
    epochs=100000
    batch_size=32
    import os
    from scipy.io import loadmat
    img_rows, img_cols = 28, 20
    ff = loadmat('../data/freyface/frey_rawface.mat', squeeze_me=True, struct_as_record=False)
    x = ff["ff"].T.reshape((-1, img_rows, img_cols))/255
    model = mvb.model_vae_bayes(dim_z_hidden=200, dim_z=10, input_shape=x.shape[1:], dim_x_hidden=200)
    optimizer = tf.keras.optimizers.Adamax()
    options = {'length': 100, 'seed': 0}
    for epoch in range(epochs):
        loss_func = ec.elbo_calculator().get_loss_func()
        loss = train.train_one_epoch(model, x, optimizer, loss_func, y=None, batch_size=batch_size, options=options)
        if (epoch+1) % 1 == 0:
            print('epoch %s: loss = %s' %(epoch+1, -loss.numpy()))

def load_frey(shape=None, options=None):
    if shape:
        raise ValueError('shape should be None in load_frey')
    img_rows, img_cols = 28, 20
    ff = loadmat('../data/freyface/frey_rawface.mat', squeeze_me=True, struct_as_record=False)
    return ff["ff"].T.reshape((-1, img_rows, img_cols))/255

def load_normal(shape=None, options=None):
    if shape is None:
        shape=(1000,1)
    return np.random.normal(0, options.get('scale', 1), shape)
    #return np.random.normal(0, 1, shape) + np.random.normal(0, 1, shape)            
            
test_cases = \
{
    'frey1':
    {
        'epochs': 10000,
        'batch_size': 32,
        'input_shape': None,
        'model_dims': (200,10,200),
        'dataset_loader': load_frey,
        'model_class': mvb.model_vae_bayes,
        'options':
        {
            'length': 100,
            'seed': 0
        },
        'use_sgd':True
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
    return model

###
### main
###
parser = argparse.ArgumentParser()
parser.add_argument('--test', help='test case', required=True)
args = parser.parse_args()
run_test(args.test)