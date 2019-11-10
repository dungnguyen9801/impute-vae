import sys
import numpy as np
import tensorflow as tf
sys.path.append('../')
sys.path.append('../../')
from shared import utils
import model_hi_vae as mhv

def test_data_transform():
    column_types = [-1, 1, 0, 3, 2]
    x0 = np.exp(np.array([[1],[2],[3]]))
    x1 = np.array([[4],[5],[6]])
    x2 = np.exp(np.array([[10],[11],[12]]))
    x3 = np.array([[0],[1],[2]])
    x4 = np.array([[1],[0],[1]])
    x = tf.concat([x0,x1,x2,x3,x4], axis=-1).numpy()
    y = np.array([
        [1, 4, 10, 1, 0, 0, 0, 1],
        [2, 5, 11, 0, 1, 0, 1, 0],
        [3, 6, 12, 0, 0, 1, 0, 1]])
    z = utils.transform_data(x, column_types).astype(np.int32)
    assert np.min(y==z)

#to finish
def test_hi_vae_encoder_gaussian():
    column_types = [1]
    x = np.array([[1],[2],[3]], dtype='float32')
    input_dim = 1
    hidden_dim = 1
    latent_dim = 1
    s_dim = 1
    model = mhv.model_hi_vae(
        input_dim,
        hidden_dim,
        latent_dim,
        s_dim,
        column_types)
    hidden_weights = (np.array([[1.0]]), np.array([2.0]))
    mu_weights = (np.array([[3.0],[4.0]]), np.array([5.0]))
    sigma_weights = (np.array([[6.0],[7.0]]), np.array([8.0]))
    s_probs_weights = (np.array([[9.0]]), np.array([10.0])) 
    model.encoder.set_weights([*hidden_weights, *mu_weights, *sigma_weights, *s_probs_weights])
    # print (model.encoder(x))
    # assert False