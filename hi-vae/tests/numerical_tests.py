import sys
import numpy as np
import tensorflow as tf
sys.path.append('../')
sys.path.append('../../')
from utils import hi_vae_utils as utils
import hi_vae_functions as hvf
import model_hi_vae as mhv

def test_x_additive_elbo_loss():
    tf.keras.backend.clear_session()
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
    x = utils.transform_data_hi_vae(x, column_types).astype(np.float32)
    batch = len(x0)

    hidden_x_dim = 4
    z_dim = 3
    s_dim = 2

    miss_list = np.ones(tf.shape(x))

    model = mhv.model_hi_vae(
        x.shape[-1],
        hidden_x_dim,
        z_dim,
        s_dim,
        column_types)

    #init network weights
    np.random.seed(1)
    weights = []
    for w in model.endecoder.get_weights():
        weights.append(np.random.normal(0,1,size=w.shape))
    model.endecoder.set_weights(weights)

    sample_length = 1000
    eps = np.random.normal(0,1, size=(
        sample_length,
        s_dim,
        batch,
        z_dim))
    losses = []
    for ids in ([0], [1], [2], [0,1,2]):
        x_batch = x[ids]
        miss_batch = miss_list[ids]
        x_norm, x_avg, x_std = hvf.get_trivial_batch_normalization(
            x_batch,
            miss_batch,
            column_types)
        _, _, _, _, _, _, elbo_loss = \
            model.endecoder([x_batch, x_norm, x_avg, x_std, eps[:,:,ids,:]])
        losses.append(elbo_loss.numpy())
    assert(abs(1.0 - (losses[0] + losses[1] + losses[2])/3/losses[3]) < 0.05)

def test_eps_additive_elbo_loss():
    tf.keras.backend.clear_session()
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
    x = utils.transform_data_hi_vae(x, column_types).astype(np.float32)
    batch = len(x0)

    hidden_x_dim = 4
    z_dim = 3
    s_dim = 2

    miss_list = np.ones(tf.shape(x))

    model = mhv.model_hi_vae(
        x.shape[-1],
        hidden_x_dim,
        z_dim,
        s_dim,
        column_types)

    #init network weights
    np.random.seed(1)
    weights = []
    for w in model.endecoder.get_weights():
        weights.append(np.random.normal(0,1,size=w.shape))
    model.endecoder.set_weights(weights)

    sample_length = 1000
    eps1 = np.random.normal(0,1, size=(
        sample_length,
        s_dim,
        batch,
        z_dim))
    eps2 = np.random.normal(0,1, size=(
        sample_length,
        s_dim,
        batch,
        z_dim))
    losses = []
    for eps in (eps1, eps2, np.concatenate([eps1, eps2], axis = 0)):
        x_batch = x
        miss_batch = miss_list
        x_norm, x_avg, x_std = hvf.get_trivial_batch_normalization(
            x_batch,
            miss_batch,
            column_types)
        _, _, _, _, _, _, elbo_loss = \
            model.endecoder([x_batch, x_norm, x_avg, x_std, eps])
        losses.append(elbo_loss.numpy())
    assert(abs(1.0 - (losses[0] + losses[1])/2/losses[2]) < 0.05)