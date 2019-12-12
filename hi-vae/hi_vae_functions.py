import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from utils import hi_vae_utils as utils

def get_batch_normalization(x, miss_list, column_types):
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

def get_x_hidden(graph, hidden_dim, x_norm):
    if not 'x_hidden' in graph:
        graph['x_hidden'] = keras.layers.Dense(
            hidden_dim,
            activation='tanh',
            name='x_hidden')
    return graph['x_hidden'](x_norm)

def get_s_probs(graph, x_hidden, s_dim):
    if not 's_probs_layer' in graph:
        graph['s_probs_layer'] = keras.layers.Dense(
            s_dim,
            activation='softmax',
            name='s_probs_layer')
    return graph['s_probs_layer'](x_hidden)

def attach_s_vectors(x_hidden, s_dim):
    x_s = tf.keras.backend.repeat(x_hidden,s_dim)
    id_mat = tf.stack([tf.eye(s_dim)], axis=0)
    id_mat_batch = tf.tile(id_mat, tf.stack([tf.shape(x_s)[0], 1, 1], name='stack_x'))
    x_s = tf.concat([x_s, id_mat_batch], axis=-1)
    return tf.transpose(x_s, [1,0,2])

def get_z_parameters(graph, x_s, z_dim):
    if 'mu_z' not in graph:
        graph['mu_z'] = \
            keras.layers.Dense(z_dim, activation='linear', name='mu_z')
    if 'log_sigma_z' not in graph:
        graph['log_sigma_z'] = \
            keras.layers.Dense(z_dim, activation='linear', name='log_sigma_z')
    return graph['mu_z'](x_s), graph['log_sigma_z'](x_s)

def get_z_samples(mu_z, log_sigma_z, options=None):
    sigma_z = tf.math.exp(log_sigma_z)
    if not options:
        sample_length = 100
        seed = 0
    else:
        sample_length = options['length']
        seed = options['seed']
    if seed:
        np.random.seed(seed)
    s_dim, batch, z_dim = mu_z.shape
    eps = np.random.normal(0,1, size = (sample_length, s_dim, batch, z_dim))
    return tf.transpose(eps*sigma_z + mu_z, [1,0,2,3])

def get_y_decode(graph, z_samples, input_dim):
    if graph.get('y_shared_layer', None) == None:
        graph['y_shared_layer'] = keras.layers.Dense(
            input_dim,
            activation='tanh',
            name='y_decode')
    return graph['y_shared_layer'](z_samples)

def get_x_parameters(graph, y, s_dim, beta, gamma, column_types):
    assert(len(tf.shape(y)) == 4)
    y = tf.reshape(y, [-1, tf.shape(y)[-1]])
    s_tail = tf.reshape(
        tf.keras.backend.repeat(
            tf.eye(s_dim),
            tf.shape(y)[0]//s_dim),
        [-1, s_dim])
    x_params = []
    if not 'x_params_layers' in graph:
        graph['x_params_layers'] = [None] * len(column_types)
    x_params_layers = graph['x_params_layers']
    d = 0
    for i, column in enumerate(column_types):
        type_, dim = column['type'], column['dim']
        y_i_s = tf.concat(
            [y[:,i:i+1], s_tail],
            axis=-1)
        if type_ == 'count':
            if not x_params_layers[i]:
                x_params_layers[i] = keras.layers.Dense(1)
            x_params.append(x_params_layers[i](y_i_s))
        elif type_ == 'real' or type_ == 'pos':
            if not x_params_layers[i]:
                x_params_layers[i] = (
                    keras.layers.Dense(1), 
                    keras.layers.Dense(1)
                )
            x_params.append(tf.concat(
                [x_params_layers[i][0](y_i_s) * gamma[d] + beta[d],
                x_params_layers[i][1](y_i_s) * gamma[d]],
                axis = -1,
            ))
        else:
            if not x_params_layers[i]:
                x_params_layers[i] = keras.layers.Dense(dim, activation='softmax')
            x_params.append(x_params_layers[i](y_i_s))
        d += dim
    return x_params

def get_elbo_loss(graph, z_samples, mu_z, log_sigma_z, x, miss_list, x_params, column_types):
    s_dim, sample_length, batch, z_dim = tf.shape(z_samples)
    z_samples = tf.reshape(z_samples, [-1, z_dim])
    assert(mu_z.numpy().shape == (batch, z_dim))
    sigma_z = tf.exp(log_sigma_z)

    # calculate E_{q(z|x)}log(q(z|x))

    log_q_z_x = utils.get_gaussian_densities(
        z_samples,
        mu_z,
        sigma_z)/z_samples/batch
    
    assert(log_q_z_x.numpy().shape == z_samples.shape())

    # per each s value
    log_q_z_x = tf.reduce_sum(
        tf.reshape(log_q_z_x, [s_dim, -1]),
        axis=-1)
    
    # calculate E_{q(z|x)}log(p(x|z))
    eps = 1e-5
    assert(len(x_params) == len(column_types))
    d = 0
    loss_mat = []
    for i, column in enumarate(column_types):
        type_, dim = column['type'], column['dim']
        x_d = x[:,d:d+dim]
        miss_list_d = miss_list[:,i:i+1]
        if type_ == 'count':
            log_lambda_d = x_params[i]
            assert(log_lambda_d.numpy().shape ==(s_dim * sample_length *batch, dim))
            log_poisson_loss = tf.nn.log_poisson_loss(
                x_d,
                log_lambda_d) * miss_list_d
            loss_mat.append(log_poisson_loss)
            assert(log_lambda_d.numpy().shape ==(s_dim * sample_length *batch, dim))
        elif type_ == 'real' or type_ == 'pos':
            mu_x_d, log_sigma_x_d = x_params[i][:,:1], x_params[i][:,1:]
            loss_mat.append(utils.get_gaussian_densities(
                x_d,
                mu_x_d,
                log_sigma_x_d) * miss_list_d
            )
        else:
            x_probs_d = x_params[i]
            assert(x_probs_d.numpy().shape == 
                (s_dim * sample_length *batch, dim))
            loss_mat.append(
                tf.math.log(
                    tf.reduce_mean(
                        tf.clip_by_value(
                            x_d * x_probs_d,
                            clip_value_min=eps,
                            clip_value_max=None))) * miss_list_d)
        d += dim

    loss_mat = tf.concat(loss_mat, axis=-1)/sample_length/batch
    log_p_x_z = tf.math.reduce_sum(
        tf.reshape(loss_mat, [s_dim,-1]),
        axis=-1)

    # calculate E_{q(z|x)}log(p(z))
    z_s = tf.keras.backend.repeat(z_samples, s_dim)
    if 's_component_means' not in graph:
        graph['s_component_means'] = tf.Variable(
            np.random.normal(0,1,size=(s_dim,z_dim)))

    log_p_z = utils.get_gaussian_densities(
        z_s,
        graph['s_component_means'],
        1)

    log_p_z = tf.reshape(log_p_z, [s_dim, -1])/sample_length/batch/s_dim
    log_p_z = tf.math.reduce_sum(log_p_z, axis=-1)
    return tf.math.reduce_sum(tf.matmul(s_probs, log_p_z))

def get_hi_vae_encoder(graph, column_types, input_dim, hidden_dim, z_dim, s_dim, options=None):
    x = keras.layers.Input(shape=(input_dim,))
    miss_list = keras.layers.Input(shape=(input_dim,))
    x_norm, _, x_avg, x_std = get_batch_normalization(x, miss_list, column_types)
    x_hidden = get_x_hidden(graph, hidden_dim, x_norm)
    s_probs = get_s_probs(graph, x_hidden,s_dim)
    x_s = attach_s_vectors(x_hidden, s_dim) 
    mu_z, log_sigma_z = get_z_parameters(graph, x_s, z_dim)
    z_samples = get_z_samples(graph, mu_z, log_sigma_z)

    #decoder
    y_decode = get_y_decode(graph, z_samples, input_dim)
    x_params = get_predict_parameters(graph, y_decode, s_dim, x_avg, x_std)

    return keras.models.Model(
        inputs=x_miss_list,
        outputs=(s_probs, mu_z, log_sigma_z, x_avg, x_std))