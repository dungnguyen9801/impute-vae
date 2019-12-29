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
model = mhv.model_hi_vae(
    x.shape[-1],
    test['hidden_x_dim'],
    test['z_dim'],
    test['s_dim'],
    column_types
)
batch_size = test['batch_size']

# x_norm = tf.keras.layers.Input(shape=(14,))
# mu_z_layer = tf.keras.layers.Dense(
#     test['z_dim'],
#     activation='sigmoid',
#     name='mu_z_test_main')
# # mu_z = mu_z_layer(x_norm)
# # test_loss = tf.math.reduce_sum(mu_z**2)
# model = tf.keras.models.Model(
#     inputs=x_norm,
#     outputs=tf.reduce_sum(mu_z_layer(x_norm))
# )

optimizer = tf.keras.optimizers.Adamax()
for iter_ in range(100000):
    with tf.GradientTape() as tape:
        ids = np.random.choice(x.shape[0], size=batch_size, replace=False)
        x_batch = x[ids]
        miss_batch = miss_list[ids]
        x_norm, x_avg, x_std = hvf.get_batch_normalization(
            x_batch,
            miss_batch,
            column_types)
        # mu_z, log_sigma_z, x_params, elbo_loss, =\
        #     model.endecoder([x_batch, x_norm,x_avg, x_std])
        test_loss = model.endecoder(x_norm)
        print('iter =%s, loss = %s' %(iter_, test_loss))
        # if (np.isnan(test_loss.numpy())):
        #     print('iter =%s, loss = %s,x_std =%s, norm=%s' %(iter_, test_loss,x_std,x_norm))
        #     exit()
        variables = model.endecoder.trainable_variables
        gradients = tape.gradient(test_loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))
