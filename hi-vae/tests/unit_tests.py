import sys
import numpy as np
import tensorflow as tf
sys.path.append('../')
sys.path.append('../../')
from shared import utils
import hi_vae_functions as hvf
import pytest

def test_batch_normalization_no_miss():
    column_types =[
        {'type': 'cat', 'dim':3},
        {'type': 'real', 'dim':1},
        {'type': 'count', 'dim':1},
        {'type': 'pos', 'dim':1},
        {'type': 'cat', 'dim':2}]
    x0 = np.array([[0],[1],[2]])
    x1 = np.array([[4],[5],[6]])
    x2 = np.exp(np.array([[10],[11],[12]]))
    x3 = np.exp(np.array([[1],[2],[3]]))
    x4 = np.array([[1],[0],[1]])
    x = np.concatenate([x0,x1,x2,x3,x4], axis=-1)
    y = np.array([
        [1, 0, 0, 4, 10, 1, 0, 1],
        [0, 1, 0, 5, 11, 2, 1, 0],
        [0, 0, 1, 6, 12, 3, 0, 1]])
    x = utils.transform_data_hi_vae(x, column_types).astype(np.int32)
    assert(np.min(x==y))

    miss_list = utils.transform_data_miss_list(np.ones((3,5)), column_types)
    x_norm, x_avg, x_std = hvf.get_batch_normalization(x, miss_list,column_types)

    x_norm_expected = np.array([
        [1., 0., 0., -1.22474487, 10., -1.224744870, 0., 1.],
        [0., 1., 0., 0., 11., 0., 1., 0.],
        [0., 0., 1., 1.22474487, 12., 1.22474487, 0., 1.]])
    
    x_avg_expected = np.array([0,0,0,5,0,2.,0,0])
    x_std_expected = np.array([1,1,1,0.81649658,1,0.81649658,1,1])

    eps = 1e-5
    assert(np.max(np.abs(x_norm - x_norm_expected)) < eps)
    assert(np.max(np.abs(x_avg - x_avg_expected)) < eps)
    assert(np.max(np.abs(x_std - x_std_expected)) < eps)

def test_batch_normalization_with_miss():
    column_types =[
        {'type': 'cat', 'dim':3},
        {'type': 'real', 'dim':1},
        {'type': 'count', 'dim':1},
        {'type': 'pos', 'dim':1},
        {'type': 'cat', 'dim':2}]
    x0 = np.array([[0],[1],[2]])
    x1 = np.array([[4],[5],[6]])
    x2 = np.exp(np.array([[10],[11],[12]]))
    x3 = np.exp(np.array([[1],[2],[3]]))
    x4 = np.array([[1],[0],[1]])
    x = np.concatenate([x0,x1,x2,x3,x4], axis=-1)
    y = np.array([
        [1, 0, 0, 4, 10, 1, 0, 1],
        [0, 1, 0, 5, 11, 2, 1, 0],
        [0, 0, 1, 6, 12, 3, 0, 1]])
    x = utils.transform_data_hi_vae(x, column_types).astype(np.int32)
    assert(np.min(x==y))

    miss_list = np.array([
        [1, 0, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [1, 1, 1, 0, 0]]
    )
    miss_list = utils.transform_data_miss_list(miss_list, column_types)
    x_norm, x_avg, x_std = hvf.get_batch_normalization(x, miss_list,column_types)

    x_norm_expected = np.array([
        [1., 0., 0., 0., 10., -1., 0., 0.],
        [0., 0., 0., -1., 11., 1., 0., 0.],
        [0., 0., 1., 1., 12., 0., 0., 0.]])
    
    x_avg_expected = np.array([0., 0., 0., 5.5, 0., 1.5, 0., 0.])
    x_std_expected = np.array([1., 1., 1., 0.5, 1., 0.5, 1., 1.])

    eps = 1e-5
    assert(np.max(np.abs(x_norm - x_norm_expected)) < eps)
    assert(np.max(np.abs(x_avg - x_avg_expected)) < eps)
    assert(np.max(np.abs(x_std - x_std_expected)) < eps)


def test_hidden():
    graph = {}
    hidden_dim = 3
    x_norm = np.array([[1.,2,3,4,5]]).astype(np.float32)
    x_dim = x_norm.shape[-1]
    w = np.random.normal(0,1, size=(x_dim, hidden_dim)).astype(np.float32)
    b = np.random.normal(0,1, size=hidden_dim).astype(np.float32)
    graph['x_hidden'] = tf.keras.layers.Dense(hidden_dim, activation='tanh', name='x_hidden')
    layer = graph['x_hidden']
    layer(x_norm)
    layer.set_weights([w,b])
    x_hidden = hvf.get_x_hidden(graph, hidden_dim, x_norm)
    x_hidden_expected = tf.nn.tanh(tf.matmul(x_norm, w) + b).numpy()
    eps = 1e-5
    assert(np.max(np.abs(x_hidden_expected - x_hidden)) < eps)

def get_s_probs(graph, x_hidden, s_dim):
    if not 's_probs' in graph:
        graph['s_probs_layer'] = keras.layers.Dense(
            s_dim,
            activation='softmax',
            name='s_probs_layer')
    return graph['s_probs_layer'](x_hidden)

def test_s_probs():
    graph = {}
    s_dim = 2
    x_hidden = np.array([[1.,3,5]]).astype(np.float32)
    hidden_dim = x_hidden.shape[-1]
    w = np.random.normal(0,1, size=(hidden_dim, s_dim)).astype(np.float32)
    b = np.random.normal(0,1, size=s_dim).astype(np.float32)
    graph['s_probs_layer'] = tf.keras.layers.Dense(
        s_dim, 
        activation='softmax',
        name='s_probs_layer')
    layer = graph['s_probs_layer']
    layer(x_hidden)
    layer.set_weights([w,b])
    s_probs = hvf.get_s_probs(graph, x_hidden, s_dim)
    s_probs_expected = tf.nn.softmax(tf.matmul(x_hidden, w) + b).numpy()
    eps = 1e-5
    assert(np.max(np.abs(s_probs_expected - s_probs)) < eps)

def test_attach_s_vectors_1():
    x_hidden = tf.constant([[4.,2,3]])
    s_dim = 3
    x_s = hvf.attach_s_vectors(x_hidden, s_dim).numpy()
    x_s_expected = np.array([
        [[4.,2,3,1,0,0]], 
        [[4.,2,3,0,1,0]],
        [[4.,2,3,0,0,1]]])
    assert(np.min(x_s == x_s_expected))

def test_attach_s_vectors_2():
    x_hidden = tf.constant([[3.,5],[2,6]])
    s_dim = 2
    x_s = hvf.attach_s_vectors(x_hidden, s_dim).numpy()
    x_s_expected = np.array([
        [[3.,5,1,0], [2.,6,1,0]],
        [[3.,5,0,1], [2.,6,0,1]]])
    assert(np.min(x_s == x_s_expected))

def test_get_z_parameters():
    graph = {}
    z_dim = 2
    graph['mu_z'] = \
        tf.keras.layers.Dense(z_dim, activation='linear', name='mu_z')
    graph['log_sigma_z'] = \
        tf.keras.layers.Dense(z_dim, activation='linear', name='log_sigma_z')
    x_s = np.array([
        [[3.,5,2,1,0], [2.,7,6,1,0]],
        [[3.,5,2,0,1], [2.,7,6,0,1]]])
    s_dim, batch, _ = x_s.shape
    mu_z, log_sigma_z = hvf.get_z_parameters(graph, x_s, z_dim)
    assert(mu_z.numpy().shape == (s_dim, batch, z_dim))
    assert(log_sigma_z.numpy().shape == (s_dim, batch, z_dim))

def test_get_z_samples():
    s_dim, batch, z_dim = 2,5,3
    mu_z = np.random.normal(5,1, (s_dim, batch,z_dim))
    log_sigma_z = np.random.normal(np.log(1),0.1, (s_dim, batch,z_dim))
    L = 10000
    options = {'length' : L,'seed' : 1}
    z_samples = hvf.get_z_samples(mu_z, log_sigma_z, options)
    assert(z_samples.numpy().shape == (s_dim, L, batch, z_dim))
    z_samples_mean = tf.math.reduce_mean(z_samples, axis=1).numpy()
    z_samples_std = tf.math.reduce_std(z_samples, axis=1).numpy()
    eps = 0.05
    assert(np.max(np.abs(z_samples_mean-mu_z)) < eps)
    assert(np.max(np.abs(z_samples_std-np.exp(log_sigma_z))) < eps)

def test_get_y_decode():
    graph = {}
    input_dim = 4
    s_dim, L, batch, z_dim = 2,4,5,3
    z_samples = np.random.normal(2,1,size=(s_dim,L,batch,z_dim))\
        .astype(np.float32)
    if graph.get('y_shared_layer', None) == None:
        graph['y_shared_layer'] = tf.keras.layers.Dense(input_dim, name='y_decode')
    layer = graph['y_shared_layer']
    y_decode = hvf.get_y_decode(graph, z_samples, input_dim).numpy()
    assert(y_decode.shape == ((s_dim,L,batch,input_dim)))

def test_get_y_decode_2():
    graph = {}
    input_dim = 4
    s_dim, L, batch, z_dim = 2,4,5,3
    z_samples = np.random.normal(10000,1,size=(s_dim,L,batch,z_dim))\
        .astype(np.float32)
    y_decode = hvf.get_y_decode(graph, z_samples, input_dim).numpy()
    assert(np.max(y_decode) <= 1.)
    assert(np.min(y_decode) >= -1.)
    
