import tensorflow as tf
import numpy as np
from . import pixel_dcn


"""
This module provides some short functions to reduce code volume
"""


def pixel_dcl(inputs, out_num, kernel_size, scope, data_type='2D', action='add'):
    if data_type == '2D':
        outs = pixel_dcn.pixel_dcl(inputs, out_num, kernel_size, scope, None)
    else:
        outs = pixel_dcn.pixel_dcl3d(inputs, out_num, kernel_size, scope, action, None)
    return tf.contrib.layers.batch_norm(
        outs, decay=0.9, epsilon=1e-5, activation_fn=tf.nn.relu,
        updates_collections=None, scope=scope+'/batch_norm')

def ipixel_dcl(inputs, out_num, kernel_size, scope, data_type='2D', action='add'):
    if data_type == '2D':
        outs = pixel_dcn.ipixel_dcl(inputs, out_num, kernel_size, scope, None)
    else:
        outs = pixel_dcn.ipixel_dcl3d(
            inputs, out_num, kernel_size, scope, action, None)
    return tf.contrib.layers.batch_norm(
        outs, decay=0.9, epsilon=1e-5, activation_fn=tf.nn.relu,
        updates_collections=None, scope=scope+'/batch_norm')

def ipixel_cl(inputs, out_num, kernel_size, scope, data_type='2D'):
    # only support 2d
    outputs = pixel_dcn.ipixel_cl(inputs, out_num, kernel_size, scope, None)
    return tf.contrib.layers.batch_norm(
        outputs, decay=0.9, epsilon=1e-5, activation_fn=tf.nn.relu,
        updates_collections=None, scope=scope+'/batch_norm')

def batch_norm(inputs, scope, is_train = True, ac_fn = tf.nn.relu):
    return tf.contrib.layers.batch_norm(
        inputs, decay=0.9, scale=True, activation_fn=ac_fn,
        updates_collections=None, epsilon=1.1e-5, is_training=is_train,
        scope = scope+'/batch_norm')

def conv(inputs, out_num, kernel_size, scope, data_type='2D', stride=1):
    if data_type == '2D':
        outs = tf.layers.conv2d(
            inputs, out_num, kernel_size, padding='same', name=scope+'/conv',
            stride=stride, kernel_initializer=tf.truncated_normal_initializer)
    else:
        shape = list(kernel_size) + [inputs.shape[-1].value, out_num]
        weights = tf.get_variable(
            scope+'/conv/weights', shape,
            initializer=tf.truncated_normal_initializer())
        outs = tf.nn.conv3d(
            inputs, weights, (1, stride, stride, stride, 1), padding='SAME',
            name=scope+'/conv')
    return outs

def leaky_relu(inputs, name='lrelu'):
    return tf.maximum(inputs, 0.2*inputs, name=name)

def deconv(inputs, out_num, kernel_size, scope, data_type='2D', **kws):
    if data_type == '2D':
        outs = tf.layers.conv2d_transpose(
            inputs, out_num, kernel_size, (2, 2), padding='same', name=scope,
            kernel_initializer=tf.truncated_normal_initializer)
    else:
        shape = list(kernel_size) + [out_num, out_num]
        input_shape = inputs.shape.as_list()
        out_shape = [input_shape[0]] + \
            list(map(lambda x: x*2, input_shape[1:-1])) + [out_num]
        weights = tf.get_variable(
            scope+'/deconv/weights', shape,
            initializer=tf.truncated_normal_initializer())
        outs = tf.nn.conv3d_transpose(
            inputs, weights, out_shape, (1, 2, 2, 2, 1), name=scope+'/deconv')
    return outs


def pool(inputs, kernel_size, scope, data_type='2D'):
    if data_type == '2D':
        return tf.layers.max_pooling2d(inputs, kernel_size, (2, 2), name=scope)
    return tf.layers.max_pooling3d(inputs, kernel_size, (2, 2, 2), name=scope)

def ssim_loss(img1, img2, size=11, sigma=1.5):
    img1s = tf.split(img1, img1.shape[-1].value, axis=4)
    img2s = tf.split(img2, img2.shape[-1].value, axis=4)
    window = fspecial_gauss(size, sigma)
    K1 = 0.01
    K2 = 0.03
    L = 1
    C1 = (K1 * L)**2
    C2 = (K2 * L)**2
    values = []
    for index in range(img1.shape[-1].value):
        mu1 = tf.nn.conv3d(img1s[index], window, strides=[1,1,1,1,1], padding='VALID')
        mu2 = tf.nn.conv3d(img2s[index], window, strides=[1,1,1,1,1], padding='VALID')
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = tf.nn.conv3d(img1s[index]*img1s[index], window, strides=[1,1,1,1,1], padding='VALID') - mu1_sq
        sigma2_sq = tf.nn.conv3d(img2s[index]*img2s[index], window, strides=[1,1,1,1,1], padding='VALID') - mu2_sq
        sigma12 = tf.nn.conv3d(img1s[index]*img2s[index], window, strides=[1,1,1,1,1], padding='VALID') - mu1_mu2
        value = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2))
        values.append(tf.reduce_mean(value))
    return (1-tf.reduce_mean(values))/2

def fspecial_gauss(size, sigma):
    x_data, y_data, z_data = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    x_data = np.expand_dims(x_data, axis=-1)
    x_data = np.expand_dims(x_data, axis=-1)
    y_data = np.expand_dims(y_data, axis=-1)
    y_data = np.expand_dims(y_data, axis=-1)
    z_data = np.expand_dims(z_data, axis=-1)
    z_data = np.expand_dims(z_data, axis=-1)
    x = tf.constant(x_data, dtype=tf.float32)
    y = tf.constant(y_data, dtype=tf.float32)
    z = tf.constant(y_data, dtype=tf.float32)
    g = tf.exp(-((x**2 + y**2 + z**2)/(2.0*sigma**2)))
    return g / tf.reduce_sum(g)