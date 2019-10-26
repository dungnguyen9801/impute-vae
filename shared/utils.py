import tensorflow as tf
import numpy as np

def get_gaussian_densities(zs, mu_z, sigma_z):
    return -(zs - mu_z)**2/(2*sigma_z*sigma_z) - 0.5*tf.math.log(2*np.pi*sigma_z*sigma_z)

# x = numpy array
def transform_data(x, column_types):
    eps = 0.0001
    y = np.array(x).astype(np.float32)
    fields = []
    for i in range(len(column_types)):
        if column_types[i] == 1:
            fields.append(y[:,i:i+1])
        elif column_types[i] == -1 or column_types[i] == 0:
            fields.append(np.log(y[:,i:i+1]+eps))
        else:
            v = y[:,i:i+1].astype(np.int32)
            fields.append(
                tf.one_hot(
                    tf.reshape(v, (len(y),)),
                    column_types[i]).numpy())
    return tf.concat(fields, axis=-1).numpy()


