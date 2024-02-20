#!/usr/bin/python
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from params import Params as param

def ResUnit(inputs, filters, kernel_size, strides, scope, reuse=None):
    np.random.seed(1337)
    with tf.variable_scope(scope, reuse=reuse):
        outputs = tf.nn.relu(inputs, name='relu')
        outputs = tf.layers.conv2d(outputs, filters, kernel_size, strides, padding='SAME', name='conv1', reuse=reuse)

        outputs = tf.nn.relu(outputs, name='relu')
        outputs = tf.layers.conv2d(outputs, filters, kernel_size, strides, padding='SAME', name='conv2', reuse=reuse)

        outputs = tf.add(outputs, inputs)
        return outputs
    
def ResNet(inputs, filters, kernel_size, repeats, scope, reuse=None):
    np.random.seed(1337)
    with tf.variable_scope(scope, reuse=reuse):
            for layer_id in range(repeats):
                inputs = ResUnit(inputs, filters, kernel_size, (1,1), "reslayer_{}".format(layer_id), reuse)
            outputs = tf.nn.relu(inputs, name='relu')
            return outputs

def ResInput(inputs, filters, kernel_size, scope, reuse=None):
    np.random.seed(1337)
    with tf.variable_scope(scope, reuse=reuse):
        outputs = tf.layers.conv2d(inputs, filters, kernel_size, strides=(1, 1), padding='SAME', name='conv_input', reuse=reuse)
        return outputs
    
def ResOutput(inputs, filters, kernel_size, scope, reuse=None):
    np.random.seed(1337)
    with tf.variable_scope(scope, reuse=reuse):
        outputs = tf.layers.conv2d(inputs, filters, kernel_size, strides=(1, 1), padding='SAME', name='conv_out', reuse=reuse)
        return outputs

def External(inputs, units):
    np.random.seed(1337)
    outputs = tf.layers.dense(inputs=inputs, units=units, activation=tf.nn.relu)
    outputs = tf.nn.relu(outputs)
    outputs = tf.layers.dense(inputs=outputs, units=15 * 16 * 1)
    outputs = tf.nn.relu(outputs)
    return outputs
    
def Fusion(closeness_output, period_output, trend_output, external_output, scope, shape):
    with tf.variable_scope(scope):
        external_output = tf.reshape(external_output, [external_output.shape[0], 15, 16, 1])
        
        outputs = tf.add(tf.add(closeness_output, period_output), trend_output)
        outputs = tf.add(outputs, external_output)
        
        return outputs

def Merge_Stream(park_stream, vd_stream, c_matrix):
    np.random.seed(1337)
    #關聯性做FC
    c_matrix = tf.reshape(c_matrix, [-1, 27 * 10])
    c_matrix = tf.layers.dense(inputs=c_matrix, units=128, activation=tf.nn.relu)
    c_matrix = tf.layers.dropout(inputs=c_matrix, rate=0.5, training=True)
    c_matrix = tf.nn.relu(c_matrix)
    c_matrix = tf.layers.dense(inputs=c_matrix, units=15 * 16 * 1)
    c_matrix = tf.nn.relu(c_matrix)
    c_matrix = tf.reshape(c_matrix, [-1, 15, 16, 1])

    #Fusion all matrix
    outputs = tf.concat([vd_stream, park_stream], 3)
    outputs = tf.add(outputs, c_matrix)
    outputs = tf.tanh(outputs)

    return outputs

def Merge_Res(park_stream, vd_stream):
    np.random.seed(1337)

    #Fusion all matrix
    outputs = tf.concat([vd_stream, park_stream], 3)
    outputs = tf.tanh(outputs)

    return outputs