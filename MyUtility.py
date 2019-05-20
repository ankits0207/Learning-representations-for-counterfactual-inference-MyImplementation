import tensorflow as tf
import numpy as np


class Parameter:
    def __init__(self, num_layers, input_dimensions, num_hidden_nodes, init_weights, p_ipm):
        self.num_layers = num_layers
        self.input_dimensions = input_dimensions
        self.num_hidden_nodes = num_hidden_nodes
        self.init_weights = init_weights
        self.p_ipm = p_ipm


class Utility:
    @staticmethod
    def get_ipm(X, p, t):
        lbound = 1e-10
        it = tf.where(t > 0)[:, 0]
        ic = tf.where(t < 1)[:, 0]
        Xc = tf.gather(X, ic)
        Xt = tf.gather(X, it)
        mean_control = tf.reduce_mean(Xc, reduction_indices=0)
        mean_treated = tf.reduce_mean(Xt, reduction_indices=0)
        c = tf.square(2 * p - 1) * 0.25
        f = tf.sign(p - 0.5)
        mmd = tf.reduce_sum(tf.square(p * mean_treated - (1 - p) * mean_control))
        mmd = f * (p - 0.5) + tf.sqrt(tf.clip_by_value(c + mmd, lbound, np.inf))
        return mmd

    @staticmethod
    def load_data(fname):
        data_in = np.load(fname)
        data = {'x': data_in['x'], 't': data_in['t'], 'yf': data_in['yf'], 'ycf': data_in['ycf']}
        data['n'] = data['x'].shape[0]
        data['dim'] = data['x'].shape[1]
        return data