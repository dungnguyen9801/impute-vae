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
    epochs=100000
    batch_size=32
    x = np.random.normal(0,2, (1000,1))
    model = mns.model_normal_simple()
    elbo_cal = ec.elbo_calculator(model, x)
    optimizer = tf.keras.optimizers.Adamax()
    options = {'length': 100, 'seed': 1}
    for epoch in range(epochs):
        loss = lambda : elbo_cal.get_elbo(options)
        optimizer.minimize(loss, model.get_trainable_variables())
        if epoch % 1000 == 0:
            print('epoch %s: loss = %s' %(epoch, loss().numpy()))
            
def test_elbo1():
    epochs=10000
    batch_size=32
    x = np.random.normal(0,1, size=(1200,1))
    model = mvb.model_vae_bayes(dim_z_hidden=1, dim_z=1, input_shape=x.shape[1:], dim_x_hidden=1)
    elbo_cal = ec.elbo_calculator(model, x)
    optimizer = tf.keras.optimizers.Adamax()
    options = {'length': 100, 'seed': 0}
    for epoch in range(epochs):
        loss = lambda : elbo_cal.get_elbo(options)
        optimizer.minimize(loss, model.get_trainable_variables())
        if epoch % 1000 == 0:
            print('epoch %s: loss = %s' %(epoch, loss().numpy()))

###
### main
###

test_elbo1b()
