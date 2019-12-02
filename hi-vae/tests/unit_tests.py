import sys
import numpy as np
import tensorflow as tf
sys.path.append('../')
sys.path.append('../../')
from shared import utils
import model_hi_vae as mhv

#to finish
def test_hi_vae_encoder():
    column_types =[
        {'type': 'pos', 'dim':1},
        {'type': 'real', 'dim':1},
        {'type': 'count', 'dim':1},
        {'type': 'cat', 'dim':3},
        {'type': 'cat', 'dim':2}]
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
    x = utils.transform_data_hi_vae(x, column_types).astype(np.int32)
    assert(np.min(x==y))

    miss_list = np.array([
        [1, 0, 1, 0, 1],
        [0, 1, 1, 0, 1],
        [1, 1, 1, 1, 0]])
    miss_list_y = np.array([
        [1, 0, 1, 0, 0, 0, 1, 1],
        [0, 1, 1, 0, 0, 0, 1, 1],
        [1, 1, 1, 1, 1, 1, 0, 0]])
    miss_list = utils.transform_data_miss_list(miss_list, column_types)

    x_miss_list = np.concat([x, miss_list], axis=-1)
    
