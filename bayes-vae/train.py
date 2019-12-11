import tensorflow as tf
import numpy as np
def train_one_epoch(model, x, optimizer, loss_func, y=None, batch_size=None, options=None):
    if not batch_size:
        batch_size = len(x)
    y_batch = None
    loss = []
    for i in range((len(x) + batch_size - 1)//batch_size):
        x_batch = x[i*batch_size: i*batch_size + batch_size]
        if y is not None:
            y_batch = y[i*batch_size: i*batch_size + batch_size]
        loss_callable = lambda : loss_func(model, x_batch, y_batch, options)
        optimizer.minimize(loss_callable, model.get_trainable_variables())
        loss.append(loss_callable())
    return tf.reduce_mean(loss)

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