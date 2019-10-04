import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import copy

tf.keras.backend.set_floatx('float32')

def get_gaussian_densities(zs, mu_z, sigma_z):
    return -(zs - mu_z)**2/(2*sigma_z*sigma_z) - 0.5*tf.math.log(2*np.pi*sigma_z*sigma_z)

def get_model_vae_gaussian(dim_z_hidden, dim_z, input_shape, dim_x_hidden):
    flatten_encode = keras.layers.Flatten()
    dense_encode = keras.layers.Dense(dim_z_hidden, activation='tanh')
    mu_encode = keras.layers.Dense(dim_z, activation='linear')
    log_sigma_encode = keras.layers.Dense(dim_z, activation='linear')
    inputs_encode = keras.layers.Input(shape=(*input_shape,))
    encoder=keras.models.Model(
        inputs=inputs_encode,
        outputs=(
            mu_encode(dense_encode(flatten_encode(inputs_encode))),
            log_sigma_encode(dense_encode(flatten_encode(inputs_encode)))
        )
    )
    
    dim_x = np.prod(input_shape)    
    dense_decode = keras.layers.Dense(dim_x_hidden, activation='tanh')
    mu_decode = keras.layers.Dense(dim_x, activation='linear')
    log_sigma_decode = keras.layers.Dense(dim_x, activation='linear')
    inputs_decode = keras.layers.Input(shape=(dim_z,))
    decoder=keras.models.Model(
        inputs=inputs_decode,
        outputs=(
            mu_decode(dense_decode((inputs_decode))),
            log_sigma_decode(dense_decode((inputs_decode)))
        )
    )
    
    return encoder, decoder

def abstract_elbo(x, z_generators, func_log_p_z_theta, func_log_q_z_x, func_log_p_x_z, options):
    zs = z_generators(x, options)
    return -(tf.math.reduce_sum(func_log_p_z_theta(zs)) 
            + tf.math.reduce_sum(func_log_p_x_z(zs, x))
            - tf.math.reduce_sum(func_log_q_z_x(zs, x)))/len(zs)

def get_z_gaussian_generator(encoder):
    def func(x, options):
        mu_z, log_sigma_z = encoder(x)
        sigma_z = tf.math.exp(log_sigma_z)
        if not options:
            L = 100
            seed = 0
        else:
            L = options['length']
            seed = options['seed']
        if seed:
            np.random.seed(seed)
        eps = np.random.normal(0,1, size = (L, x.shape[0], mu_z.shape[1]))
        return eps* sigma_z + mu_z
    return func

def func_log_p_z_gaussian(zs):
    return -zs**2/2 - 0.5 * tf.math.log(2*np.pi)

def get_func_log_q_z_x_gaussian(encoder):
    def func(zs, x):
        mu_z, log_sigma_z = encoder(x)
        sigma_z = tf.math.exp(log_sigma_z)
        return get_gaussian_densities(zs, mu_z, sigma_z)
    return func

def get_func_log_p_x_z_gaussian(decoder):
    def func(zs, x):
        mu_x, log_sigma_x = decoder(tf.reshape(zs,(-1, zs.shape[2])))
        sigma_x = tf.math.exp(log_sigma_x)
        mu_x = tf.reshape(mu_x, (-1, *x.shape))
        sigma_x = tf.reshape(sigma_x,(-1, *x.shape))
        return get_gaussian_densities(x, mu_x, sigma_x)
    return func

def new_elbo(x,encoder, decoder, L=100, seed=0):
    z_generators =  get_z_gaussian_generator(encoder)
    func_log_p_z_theta = func_log_p_z_gaussian
    func_log_q_z_x = get_func_log_q_z_x_gaussian(encoder)
    func_log_p_x_z = get_func_log_p_x_z_gaussian(decoder)
    options = {'length': L, 'seed':seed}
    return abstract_elbo(x, z_generators, func_log_p_z_theta, func_log_q_z_x, func_log_p_x_z, options)

def elbo(x,encoder, decoder, L=100, seed=0):
    
    batch = x.shape[0]
    mu_z, log_sigma_z = encoder(x)
    
    dim = mu_z.shape[1]
    
    if seed:
        np.random.seed(seed)
    eps = np.random.normal(0, 1, size = (L, batch, dim))
    
    zs = tf.reshape(eps *tf.math.exp(log_sigma_z) + mu_z, (-1, dim))
    mu_x, log_sigma_x = decoder(zs) # (L * batch, dim_x)
    mu_x = tf.reshape(mu_x, (L, batch, -1))
    log_sigma_x = tf.reshape(log_sigma_x, (L, batch, -1))
    
    minus_log_q = eps**2/2 + log_sigma_z + 0.5*tf.math.log(2*np.pi)
    log_p = -(tf.dtypes.cast(tf.reshape(x, (batch, -1)), tf.float32)-mu_x)**2/(2 * tf.math.exp(2*log_sigma_x)) -\
        log_sigma_x - 0.5*tf.math.log(2*np.pi)
    log_pz = -zs**2/2 - 0.5*tf.math.log(2*np.pi)
    return -(tf.math.reduce_sum(log_p) + tf.math.reduce_sum(minus_log_q) + tf.math.reduce_sum(log_pz))/L

def test_elbo():
    epochs=1000
    batch_size=32
    x = np.random.normal(0,1, size=(120,3,2))
    encoder, decoder = get_model_vae_gaussian(4,2,x.shape[1:],3)
    optimizer = tf.keras.optimizers.Adamax()
    assert (abs(elbo(x, encoder, decoder, L=1000,seed=1)-new_elbo(x, encoder, decoder, L=1000,seed=1)).numpy()) < 0.01
    
    for epoch in range(epochs):
        optimizer.minimize(lambda : elbo(x ,encoder, decoder), [encoder.trainable_variables, decoder.trainable_variables])
        print(elbo(x ,encoder, decoder))
