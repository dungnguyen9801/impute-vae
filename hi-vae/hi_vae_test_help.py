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
import os
import utils
from scipy.io import loadmat
import csv

def read_data(data, column_types):
    data_complete = []
    for i in range(np.shape(data)[1]):
        
        if column_types[i]['type'] == 'cat':
            #Get categories
            cat_data = [int(x) for x in data[:,i]]
            _, indexes = np.unique(cat_data,return_inverse=True)
            #Transform categories to a vector of 0:n_categories
            new_categories = np.arange(int(column_types[i]['dim']))
            cat_data = new_categories[indexes]
            #Create one hot encoding for the categories
            aux = np.zeros([np.shape(data)[0],len(new_categories)])
            aux[np.arange(np.shape(data)[0]),cat_data] = 1
            data_complete.append(aux)
            
        elif column_types[i]['type'] == 'ordinal':
            #Get categories
            cat_data = [int(x) for x in data[:,i]]
            _, indexes = np.unique(cat_data,return_inverse=True)
            #Transform categories to a vector of 0:n_categories
            new_categories = np.arange(int(column_types[i]['dim']))
            cat_data = new_categories[indexes]
            #Create thermometer encoding for the categories
            aux = np.zeros([np.shape(data)[0],1+len(new_categories)])
            aux[:,0] = 1
            aux[np.arange(np.shape(data)[0]),1+cat_data] = -1
            aux = np.cumsum(aux,1)
            data_complete.append(aux[:,:-1])
            
        elif column_types[i]['type'] == 'count':
            if np.min(data[:,i]) == 0:
                aux = data[:,i] + 1
                data_complete.append(np.transpose([aux]))
            else:
                data_complete.append(np.transpose([data[:,i]]))
            
            
            
        else:
            data_complete.append(np.transpose([data[:,i]]))
                    
    return np.concatenate(data_complete,1)

def hi_vae_random_data_load(test_case=None):
    rows = 1000
    column_types=[
        {'type': 'real', 'dim':1},
        {'type': 'real', 'dim':1},
        {'type': 'count', 'dim':1},
        {'type': 'cat', 'dim':3},
        {'type': 'cat', 'dim':2},
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
    with open(test_case['data_type_file']) as f:
        column_types = [{k: v for k, v in row.items()}
        for row in csv.DictReader(f, skipinitialspace=True)]
    for i in range(len(column_types)):
        column_types[i]['dim'] = int(column_types[i]['dim'])
    with open(test_case['data_file'], 'r') as f:
        data = [[float(x) for x in rec] for rec in csv.reader(f, delimiter=',')]
        data = np.array(data)
    miss_mask = np.ones(data.shape)
    with open(test_case['miss_file'], 'r') as f:
        missing_positions = [[int(x) for x in rec] for rec in csv.reader(f, delimiter=',')]
        missing_positions = np.array(missing_positions)
        miss_mask[missing_positions[:,0]-1,missing_positions[:,1]-1] = 0
    miss_list = utils.transform_data_miss_list(
        miss_mask.astype(np.int32),
        column_types)
    return tf.concate(
        [read_data(data, column_types), miss_list],
        axis = -1).numpy(), column_types

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
        'data_type_file': '../../data/wine/data_types.csv',
        'miss_file': '../../data/wine/Missing10_10.csv',
        'data_file': '../../data/wine/data.csv'
    }
}