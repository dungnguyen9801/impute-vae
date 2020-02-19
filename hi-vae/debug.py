import model_hi_vae as mhv
import hi_vae_functions as hvf
import hi_vae_testcases_helper as helper
import tensorflow as tf
import numpy as np
import os

def train_one_random_batch(model, x, optimizer, loss_func, y=None, batch_size=None, options=None):
    if not batch_size:
        batch_size = len(x)
    y_batch = None
    ids = np.random.choice(len(x), size=batch_size, replace=False)
    x_batch = x[ids]
    if y is not None:
        y_batch = y[ids]
    loss_callable = lambda : loss_func(model, x_batch, y_batch, options)
    optimizer.minimize(loss_callable, model.get_trainable_variables())
    loss = loss_callable()
    return tf.reduce_mean(loss)

def total_weights(model):
    ttl = .0
    for w in model.endecoder.get_weights():
        ttl += np.sum(w**2)
    return ttl

test = helper.test_cases['wine1']
options = test['options']
sample_length, s_dim, batch_size, z_dim = options['length'], test['s_dim'], test['batch_size'], test['z_dim']

iters = test['iters']
x, miss_list, column_types = test['dataset_loader'](test)
model = mhv.model_hi_vae(
    x.shape[-1],
    test['hidden_x_dim'],
    test['z_dim'],
    test['s_dim'],
    column_types
)
batch_size = test['batch_size']

np.random.seed(1)
optimizer = tf.keras.optimizers.Adamax()
weights = []
for w in model.endecoder.get_weights():
    weights.append(np.random.normal(0,1,size=w.shape))
model.endecoder.set_weights(weights)

wine_nan_path = '../weights/nan/wine.nan'
eps_nan_path = '../weights/nan/eps.nan'
model.endecoder.load_weights(wine_nan_path)
eps_flat = np.loadtxt(eps_nan_path)
thickness = s_dim * batch_size * z_dim


def point_test(mid):
    eps_flat_sub = eps_flat[mid*thickness:(mid+1)*thickness]
    eps = np.reshape(eps_flat_sub, (-1, s_dim, batch_size, z_dim))
    print(eps)
    ids = [1]
    x_batch = x[ids]
    miss_batch = miss_list[ids]
    x_norm, x_avg, x_std = hvf.get_batch_normalization(
        x_batch,
        miss_batch,
        column_types)
    mu_z, log_sigma_z, s_probs, z_samples, y_decode, x_params, elbo_loss = \
        model.endecoder([x_batch, x_norm, x_avg, x_std, eps])
    print(elbo_loss.numpy())
    print(total_weights(model))

point_test(491219)
exit()

lo, hi, bad = 0, len(eps_flat)//thickness, -1
while lo <= hi:
    mid = (lo + hi)//2
    eps_flat_sub = eps_flat[lo*thickness:(mid+1)*thickness]
    eps = np.reshape(eps_flat_sub, (-1, s_dim, batch_size, z_dim))
    ids = [1]
    x_batch = x[ids]
    miss_batch = miss_list[ids]
    x_norm, x_avg, x_std = hvf.get_batch_normalization(
        x_batch,
        miss_batch,
        column_types)
    mu_z, log_sigma_z, s_probs, z_samples, y_decode, x_params, elbo_loss = \
        model.endecoder([x_batch, x_norm, x_avg, x_std, eps])
    print(elbo_loss.numpy())
    print(total_weights(model))
    print(lo, mid, hi)
    if (np.isnan(elbo_loss.numpy())):
        ans = mid
        hi = mid-1
    else:
        lo = mid+1
print(ans)
