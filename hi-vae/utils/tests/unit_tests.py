import sys
import numpy as np
import tensorflow as tf
sys.path.append('../')
sys.path.append('../../')
from utils import hi_vae_utils as utils

def test_data_transform():
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
    z = utils.transform_data_hi_vae(x, column_types).astype(np.int32)
    assert np.min(y==z)

def test_reverse_data_transform():
    x = np.array([
        [np.log(1), 4, np.log(10), 0.7, 0.2, 0.1, 0.25, 0.75],
        [np.log(2), 5, np.log(11), 0.1, 0.6, 0.3, 0.6, 0.4],
        [np.log(3), 6, np.log(12), 0.1, 0.2, 0.7, 0.3, 0.7]])
    column_types =[
        {'type': 'pos', 'dim':1},
        {'type': 'real', 'dim':1},
        {'type': 'count', 'dim':1},
        {'type': 'cat', 'dim':3},
        {'type': 'cat', 'dim':2}]
    y = np.array([
        [1, 4, 10, 0, 1],
        [2, 5, 11, 1, 0],
        [3, 6, 12, 2, 1]])
    z = utils.reverse_transform_data_hi_vae(x, column_types).astype(np.int32)
    assert np.min(y==z)

def test_get_continuous_columns():
    column_types =[
        {'type': 'pos', 'dim':1},
        {'type': 'real', 'dim':1},
        {'type': 'count', 'dim':1},
        {'type': 'real', 'dim':1},
        {'type': 'cat', 'dim':3},
        {'type': 'pos', 'dim':1}]
    x = utils.get_continuous_columns(column_types)
    y = np.array([1,1,0,1,0,0,0,1])
    assert np.min(x==y)
    
def test_transform_miss_list():
    column_types =[
        {'type': 'pos', 'dim':1},
        {'type': 'real', 'dim':1},
        {'type': 'count', 'dim':1},
        {'type': 'cat', 'dim':3},
        {'type': 'cat', 'dim':2}]
    miss_list = np.array([
        [1, 0, 1, 0, 1],
        [0, 1, 1, 0, 1],
        [1, 1, 1, 1, 0]])
    y = np.array([
        [1, 0, 1, 0, 0, 0, 1, 1],
        [0, 1, 1, 0, 0, 0, 1, 1],
        [1, 1, 1, 1, 1, 1, 0, 0]])
    x = utils.transform_data_miss_list(miss_list, column_types)
    assert np.min(x==y)