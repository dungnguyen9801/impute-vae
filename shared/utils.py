import tensorflow as tf
import numpy as np

def get_gaussian_densities(zs, mu_z, sigma_z):
    return -(zs - mu_z)**2/(2*sigma_z*sigma_z) - 0.5*tf.math.log(2*np.pi*sigma_z*sigma_z)

# x = numpy array
def transform_data_hi_vae(x, column_types):
    eps = 0.0001
    y = np.array(x).astype(np.float32)
    fields = []
    for i in range(len(column_types)):
        if column_types[i] == 'real':
            fields.append(y[:,i:i+1])
        elif column_types[i] == 'positive' or column_types[i] == 'count':
            fields.append(np.log(y[:,i:i+1]+eps))
        else:
            v = y[:,i:i+1].astype(np.int32)
            fields.append(
                tf.one_hot(
                    tf.reshape(v, (len(y),)),
                    column_types[i]).numpy())
    return tf.concat(fields, axis=-1).numpy()

# x = numpy array
def reverse_transform_data_hi_vae(x, column_types):
    fields = []
    j = 0
    for t in column_types:
        if t == 'real':
            fields.append(x[:,j:j+1])
        elif t == 'positive' or t == 'count':
            fields.append(np.exp(x[:,j:j+1]))
        else:
            v = x[:,j:j+t]
            fields.append(np.argmax(v, axis=1).reshape((-1,1)))
    return tf.concat(fields, axis=-1).numpy()
