import model_hi_vae as mhv
import sys
import time
sys.path.append('../shared')
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import copy
import elbo_calculator as ec
import train
import argparse
import os
import utils
from scipy.io import loadmat
import csv

def hi_vae_random_data_load(test_case=None):
    rows = 1000
    column_types=[
        {'type': 'real', 'dim':1},
        {'type': 'real', 'dim':1},
        {'type': 'count', 'dim':1},
        {'type': 'categorical', 'dim':3},
        {'type': 'categorical', 'dim':2},
        {'type': 'real', 'dim':1}]
    x0 = np.random.normal(1,2,size=(rows,1))
    x1 = np.random.normal(0,1,size=(rows,1))
    x2 = np.log(np.random.poisson(1.5,size=(rows,1))+ 0.01)
    x3 = tf.one_hot(np.random.randint(0,3,size=rows),3).numpy()
    x4 = tf.one_hot(np.random.randint(0,2,size=rows),2).numpy()
    x5 = np.random.normal(0.5,1,size=(rows,1))
    x = tf.concat([x0,x1,x2,x3,x4,x5], axis=-1).numpy().astype(np.float32)
    miss_list = utils.transform_data_miss_list(
        np.random.randint(0, 2, size=(rows,6)),
        column_types)
    return x, column_types, miss_list

def hi_vae_wine_data_load(test_case):
    column_types=[{'type':'categorical', 'dim':3}] +\
         [{'type': 'positive', 'dim':1}]*12
    data = np.loadtxt(test_case['data_file'], delimiter=',').astype(np.float32)
    classes = tf.one_hot(data[:,0].astype(np.int)-1, 3).numpy()
    x = tf.concat([classes, data[:, 1:]], axis=-1).numpy().astype(np.float32)
    miss_mask = np.ones(x.shape)
    with open(test_case['miss_file'], 'r') as f:
        missing_positions = [[int(x) for x in rec] for rec in csv.reader(f, delimiter=',')]
        missing_positions = np.array(missing_positions)
        miss_mask[missing_positions[:,0]-1,missing_positions[:,1]-1] 
    miss_list = utils.transform_data_miss_list(
        miss_mask,
        column_types)
    return utils.transform_data_hi_vae(x, column_types), column_types, miss_list

test_cases = \
{
    'hi_vae1':
    {
        'epochs': 2000000,
        'batch_size': 10,
        'input_shape': None,
        'model_dims': (256,2,256),
        'dataset_loader': hi_vae_random_data_load,
        'model_class': mhv.model_hi_vae,
        'options':
        {
            'length': 1,
            'seed': 0
        },
        'use_sgd':True,
        'report_frequency':100
    },
    'hi_vae_wine':
    {
        'epochs': 2000000,
        'batch_size': 1,
        'input_shape': None,
        'model_dims': (256,2,256),
        'dataset_loader': hi_vae_wine_data_load,
        'model_class': mhv.model_hi_vae,
        'options':
        {
            'length': 10,
            'seed': 1
        },
        'use_sgd':True,
        'report_frequency':100,
        'miss_file':
    }
}