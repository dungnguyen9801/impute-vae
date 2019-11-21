import tensorflow as tf
import numpy as np

def get_gaussian_densities(zs, mu_z, sigma_z):
    return -(zs - mu_z)**2/(2*sigma_z*sigma_z) - 0.5*tf.math.log(2*np.pi*sigma_z*sigma_z)

def get_continuous_columns(column_types):
    ids = []
    for column in column_types:
        type_, dim = column['type'], column['dim']
        if type_ == 'real' or type_ =='positive':
            ids.append(1)
        else:
            ids.extend([0]*dim)
    return np.array(ids)

# x = numpy array
def transform_data_hi_vae(x, column_types):
    eps = 0.0001
    y = np.array(x).astype(np.float32)
    fields = []
    for i, column in enumerate(column_types):
        type_, dim = column['type'], column['dim']
        if type_ == 'real':
            fields.append(y[:,i:i+1])
        elif type_ == 'positive' or type_ == 'count':
            fields.append(np.log(y[:,i:i+1]+eps))
        else:
            v = y[:,i:i+1].astype(np.int32)
            fields.append(
                tf.one_hot(
                    tf.reshape(v, (len(y),)),
                    dim).numpy())
    return tf.concat(fields, axis=-1).numpy()

def transform_data_miss_list(miss_list, column_types):
    fields = []
    rows = tf.shape(miss_list)[0]
    for i, column in enumerate(column_types):
        _, dim = column['type'], column['dim']
        fields.append(tf.broadcast_to(miss_list[:,i:i+1], (rows, dim)).numpy())
    return tf.concat(fields, axis=-1).numpy().astype(np.float32)

# x = numpy array
def reverse_transform_data_hi_vae(x, column_types):
    fields = []
    j = 0
    for column in column_types:
        type_, dim = column['type'], column['dim']
        if type_ == 'real':
            fields.append(x[:,j:j+1])
        elif type_ == 'positive' or type_ == 'count':
            fields.append(np.exp(x[:,j:j+1]))
        else:
            v = x[:,j:j+dim]
            fields.append(np.argmax(v, axis=1).reshape((-1,1)))
        j += dim
    return tf.concat(fields, axis=-1).numpy()
