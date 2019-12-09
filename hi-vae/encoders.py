import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from shared import utils

def batch_normalization(x, miss_list, column_types):
    eps = 0.0001
    cont_ids = utils.get_continuous_columns(column_types)
    x_avg = []
    x_std = []
    for i in range(tf.shape(x)[1]):
        if cont_ids[i]:
            observed_indices = np.where(miss_list[:, i] == 1)
            observed = x[observed_indices, i]
            avg = np.mean(observed)
            std = np.clip(np.std(observed), eps, None)
            x_avg.append(avg)
            x_std.append(std)
        else:
            x_avg.append(0.0)
            x_std.append(1.0)
    x_avg = np.array(x_avg)
    x_std = np.array(x_std)
    x_norm = (x - x_avg)/x_std * miss_list
    return x_norm, x_avg, x_std

def hidden(graph, hidden_dim, x_norm):
    if not 'x_hidden' in graph:
        graph['x_hidden'] = keras.layers.Dense(hidden_dim, activation='tanh', name='x_hidden')
    return graph['x_hidden'](x_norm)

def s_probabilities(graph, x_hidden, s_dim):
    if not 's_probs' in graph:
        graph['s_probs_layer'] = keras.layers.Dense(s_dim, activation='softmax', name='s_probs')
    return graph['s_probs_layer'](x_hidden)

def attach_s_vectors(x_hidden, s_dim):
    x_s = tf.keras.backend.repeat(x_hidden,s_dim)
    id_mat = tf.stack([tf.eye(s_dim)], axis=0)
    id_mat_batch = tf.tile(id_mat, tf.stack([tf.shape(x_s)[0], 1, 1], name='stack_x'))
    x_s = tf.concat([x_s, id_mat_batch], axis=-1)
    return x_s

def z_parameters(graph, x_s, latent_dim):
    if 'mu_z' not in graph:
        graph['mu_z'] = \
            keras.layers.Dense(latent_dim, activation='linear', name='mu_z')
    if 'log_sigma_z' not in graph:
        graph['log_sigma_z'] = \
            keras.layers.Dense(latent_dim, activation='linear', name='log_sigma_z')
    return tf.transpose(graph['mu_z'](x_s), [1,0,2]), \
            tf.transpose(graph['log_sigma_z'](x_s), [1,0,2])

def sample_z(mu_z, log_sigma_z, options=None):
    sigma_z = tf.math.exp(log_sigma_z)
    if not options:
        L = 100
        seed = 0
    else:
        L = options['length']
        seed = options['seed']
    if seed:
        np.random.seed(seed)
    s_dim, batch, z_dim = mu_z.shape
    eps = np.random.normal(0,1, size = (L, s_dim, batch, z_dim))
    return tf.transpose(eps*sigma_z + mu_z, [1,0,2,3])

def y_decode(graph, z_samples, input_dim):
    if graph.get('y_shared_layer', None) == None:
        graph['y_shared_layer'] = keras.layers.Dense(input_dim, name='y_decode')
    return graph['y_shared_layer'](z_samples)

def component_likelihood_parameters(graph, y, s_dim):
    s_tail = tf.keras.backend.repeat(
        tf.eye(s_dim),
        tf.shape(z)[0]//s_dim)
    s_tail = tf.reshape(s_tail, (-1, s_dim))
    prop_layers = []
    for column in column_types:
        type_, dim = column['type'], column['dim']
        if type_ == 'count':
            prop_layers.append((keras.layers.Dense(1),))
        elif type_ == 'real' or type_ == 'pos':
            prop_layers.append((keras.layers.Dense(1), keras.layers.Dense(1)))
        else:
            prop_layers.append((keras.layers.Dense(dim, activation='softmax'),))
    output = []
    for d, column in enumerate(column_types):
        type_, dim = column['type'], column['dim']
        y_d_s = tf.concat(
            [y[:,d:d+1], s_tail],
            axis=-1)
        output.append(list(map(lambda f: f(y_d_s), prop_layers[d])))
        if type_ == 'real' or type_ == 'pos':
            output[d][0] = output[d][0] * gamma + beta
            output[d][1] = output[d][1] * gamma

def get_hi_vae_encoder(graph, column_types, input_dim, hidden_dim, latent_dim, s_dim, options=None):
    x = keras.layers.Input(shape=(input_dim,))
    miss_list = keras.layers.Input(shape=(input_dim,))
    x_norm, _, x_avg, x_std = batch_normalization(x, miss_list, column_types)
    x_hidden = hidden(graph, hidden_dim, x_norm)
    s_probs = s_probabilities(graph, x_hidden,s_dim)
    x_s = attach_s_vectors(x_hidden, s_dim) 
    mu_z, log_sigma_z = z_parameters(graph, x_s, latent_dim)
    z_samples = sample_z(graph, mu_z, log_sigma_z)

    #decoder
    y_decoded = y_decode(graph, z_samples, input_dim)

    return keras.models.Model(
        inputs=x_miss_list,
        outputs=(s_probs, mu_z, log_sigma_z, x_avg, x_std))

def get_hi_vae_decoder(column_types, latent_dim, s_dim):
    z = keras.layers.Input(shape=(latent_dim,))
    beta = keras.layers.Input(shape=(None,))
    gamma = keras.layers.Input(shape=(None,))
    input_dim = len(column_types)
    shared_layer = keras.layers.Dense(input_dim)
    y = shared_layer(z)
    s_component_means = tf.Variable([1.0] *s_dim, 's_component_means')
    s_tail = tf.keras.backend.repeat(
        tf.eye(s_dim),
        tf.shape(z)[0]//s_dim)
    s_tail = tf.reshape(s_tail, (-1, s_dim))
    prop_layers = []
    for column in column_types:
        type_, dim = column['type'], column['dim']
        if type_ == 'count':
            prop_layers.append((keras.layers.Dense(1),))
        elif type_ == 'real' or type_ == 'pos':
            prop_layers.append((keras.layers.Dense(1), keras.layers.Dense(1)))
        else:
            prop_layers.append((keras.layers.Dense(dim, activation='softmax'),))
    output = []
    for d, column in enumerate(column_types):
        type_, dim = column['type'], column['dim']
        y_d_s = tf.concat(
            [y[:,d:d+1], s_tail],
            axis=-1)
        output.append(list(map(lambda f: f(y_d_s), prop_layers[d])))
        if type_ == 'real' or type_ == 'pos':
            output[d][0] = output[d][0] * gamma + beta
            output[d][1] = output[d][1] * gamma
    return keras.models.Model(
        inputs=([z, beta, gamma]),
        outputs= (output, s_component_means + 0.0 * s_component_means ** 2))