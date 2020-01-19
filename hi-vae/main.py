import model_hi_vae as mhv
import hi_vae_functions as hvf
import hi_vae_testcases_helper as helper
import tensorflow as tf
import numpy as np

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

test = helper.test_cases['wine1']
iters = test['iters']
x, miss_list, column_types = test['dataset_loader'](test)
options = test['options']
model = mhv.model_hi_vae(
    x.shape[-1],
    test['hidden_x_dim'],
    test['z_dim'],
    test['s_dim'],
    column_types,
    options

)
batch_size = test['batch_size']

np.random.seed(1)
optimizer = tf.keras.optimizers.Adamax()
weights = []
for w in model.endecoder.get_weights():
    weights.append(np.random.normal(0,1,size=w.shape))
model.endecoder.set_weights(weights)

losses = [[] for i in range(3)]
for iter_ in range(1):
    for i, id in enumerate(([1,2],[2], [1],[1,2], [2], [1])):
        ids = id
        x_batch = x[ids]
        miss_batch = miss_list[ids]
        x_norm, x_avg, x_std = hvf.get_trivial_batch_normalization(
            x_batch,
            miss_batch,
            column_types)
        tf.random.set_seed(10)
        z_samples, mu_z, log_sigma_z, x_params, elbo_loss = \
            model.endecoder([x_batch, x_norm, x_avg, x_std])
        print('ids =%s, loss = %s' %(ids, elbo_loss.numpy()))
        # print('mu_z', tf.reduce_sum(mu_z**2).numpy())
        # print('log_sigma_z',tf.reduce_sum(log_sigma_z**2).numpy())
        # print('z_samples',tf.reduce_mean(z_samples**2).numpy())
        # print('eps_sum', eps_sum.numpy())
        # names = [weight.name for layer in model.endecoder.layers for weight in layer.weights]
        # weights = model.endecoder.get_weights()
        # for name, weight in zip(names, weights):
        #     if 'mu_z' in name or 'log_sigma_z' in name or 'x_hidden' in name:
        #         print(name, tf.reduce_sum(weight**2).numpy(), weight.shape)

# for iter_ in range(1, iters + 1):
#     with tf.GradientTape() as tape:
#         ids = np.random.choice(x.shape[0], size=batch_size, replace=False)
#         ids = [1]
#         x_batch = x[ids]
#         miss_batch = miss_list[ids]
#         x_norm, x_avg, x_std = hvf.get_batch_normalization(
#             x_batch,
#             miss_batch,
#             column_types)
#         z_samples, mu_z, log_sigma_z, x_params, elbo_loss, eps_sum = \
#             model.endecoder([x_batch, x_norm, x_avg, x_std])
#         if iter_ % 1 == 0:
#             print('iter =%s, loss = %s' %(iter_, elbo_loss))
#             # print('eps_sum=%s' %(eps_sum))
#         variables = model.endecoder.trainable_variables
#         gradients = tape.gradient(elbo_loss, variables)
#         optimizer.apply_gradients(zip(gradients, variables))
