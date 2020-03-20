import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import InputLayer, Conv1d, DropoutLayer, DenseLayer
from tensorlayer.iterate import minibatches

def build_network(x, is_train, layer_reuse, var_reuse, hyper_params):

    hp = hyper_params

    tl.layers.set_name_reuse(layer_reuse)
    with tf.variable_scope("model", reuse=var_reuse):

        #gamma_init = tf.random_normal_initializer(1.0, 0.02)
        gamma_init = None

        network = tl.layers.InputLayer(x, name='input')

        network = tl.layers.Conv1d(network, n_filter=hp['filters'], 
                                   filter_size=hp['kernel_size'], stride=hp['strides'], 
                                   act=tf.nn.relu, padding='SAME', name='conv_a')
        network = tl.layers.BatchNormLayer(network, act=tf.nn.relu, 
                                           gamma_init=gamma_init, is_train=is_train, name='bn_a')
        network = tl.layers.MaxPool1d(network, filter_size=hp['pool_size'], strides=hp['pool_size'], 
                                      padding='VALID', name="pool_a")
        #network = tl.layers.DropoutLayer(network, keep=keep_prob, name="drop_a")

        for i in range(6): # five layers suggested
            conv_name = "conv_{}".format(i+1)
            bn_name = "bn_{}".format(i+1)
            pool_name = "pool_{}".format(i+1)
            network = tl.layers.Conv1d(network, n_filter=hp['filters'], 
                                       filter_size=hp['kernel_size'], stride=hp['strides'], 
                                       act=tf.nn.relu, padding='SAME', name=conv_name)
            network = tl.layers.BatchNormLayer(network, act=tf.nn.relu, 
                                               gamma_init=gamma_init, is_train=is_train, name=bn_name)
            network = tl.layers.MaxPool1d(network, filter_size=pool_size, strides=pool_size, 
                                          padding='VALID', name=pool_name)
            #network = tl.layers.DropoutLayer(network, keep=keep_prob)

        # Final conv layer
        network = tl.layers.Conv1d(network, n_filter=hp['filters'], 
                                   filter_size=hp['kernel_size'], stride=hp['strides'], 
                                   act=tf.nn.relu, padding='SAME', name='conv_z')
        network = tl.layers.BatchNormLayer(network, act=tf.nn.relu, 
                                           gamma_init=gamma_init, is_train=is_train, name='bn_z')
        network = tl.layers.MaxPool1d(network, filter_size=hp['pool_size'], strides=hp['pool_size'], 
                                      padding='VALID', name='pool_z')

        network = tl.layers.FlattenLayer(network, name='flatten') 

        network = tl.layers.DenseLayer(network, n_units=hp['num_fc'], act = tf.nn.relu)  
        #network = tl.layers.DropoutLayer(network, keep=keep_prob, name='drop_z')

        network = tl.layers.DenseLayer(network, n_units=hp['num_classes'], act = tf.identity, name='output')

    return network
