import sys
import numpy as np
import tensorflow as tf
sys.path.append('../')
sys.path.append('../../')
from shared import utils
import model_hi_vae as mhv
import encoders

#to finish
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
    x_norm, x_avg, x_std = encoders.batch_normalization(x, miss_list,column_types)

    x_norm_expected = np.array([
        [1., 0., 0., -1.22474487, 10., -1.224744870, 0., 1.],
        [0., 1., 0., 0., 11., 0., 1., 0.],
        [0., 0., 1., 1.22474487, 12., 1.22474487, 0., 1.]])
    
    x_avg_expected = np.array([0,0,0,5,0,2.,0,0])
    x_std_expected = np.array([1,1,1,0.81649658,1,0.81649658,1,1])

    eps = 0.00001
    assert(np.max(np.abs(x_norm - x_norm_expected)) < eps)
    assert(np.max(np.abs(x_avg - x_avg_expected)) < eps)
    assert(np.max(np.abs(x_std - x_std_expected)) < eps)
    

    # miss_list = np.array([
    #     [1, 0, 1, 0, 1],
    #     [0, 1, 1, 0, 1],
    #     [1, 1, 1, 1, 0]])
    # miss_list_y = np.array([
    #     [1, 0, 1, 0, 0, 0, 1, 1],
    #     [0, 1, 1, 0, 0, 0, 1, 1],
    #     [1, 1, 1, 1, 1, 1, 0, 0]])