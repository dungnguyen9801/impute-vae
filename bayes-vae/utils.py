import numpy as np
import tensorflow as tf

def get_gaussian_densities(zs, mu_z, sigma_z):
    return -(zs - mu_z)**2/(2*sigma_z*sigma_z) - 0.5*tf.math.log(2*np.pi*sigma_z*sigma_z)
