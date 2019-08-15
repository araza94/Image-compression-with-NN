#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 30 12:48:04 2018

@author: akber
"""


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

learning_rate = 0.01
num_steps = 480*20
batch_size = 125
test_batch_size = 100

display_step = 480
examples_to_show = 10

# Network Parameters
num_input = 784 # MNIST data input (img shape: 28*28)
training_loss=[]
error=[]
test_loss=[]
perf = []
perf_tl=[]
rate = np.array([10**-7,10**-6,10**-5, 10**-4, 10**-3, 10**-2, 10**-1, 10**-0])
#rate=0
#rate = np.array([17,18,19,20,21,22,23])*10**-4
# tf Graph input (only pictures)
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder(tf.float32, [None, 10])
beta = tf.placeholder(tf.float32, shape=())
data_type = tf.placeholder(tf.int16, shape=())

#H = tf.constant(haarMatrix(32), name="wave", dtype=np.float32)

def weight_variable(shape, name):
    # From the mnist tutorial
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)

def haarMatrix(n, normalized=False):
    # Allow only size n of power 2
    n = 2**np.ceil(np.log2(n))
    if n > 2:
        h = haarMatrix(n / 2)
    else:
        return np.array([[1, 1], [1, -1]])

    # calculate upper haar part
    h_n = np.kron(h, [1, 1])
    # calculate lower haar part 
    if normalized:
        h_i = np.sqrt(n/2)*np.kron(np.eye(len(h)), [1, -1])
    else:
        h_i = np.kron(np.eye(len(h)), [1, -1])
    # combine parts
    h = np.vstack((h_n, h_i))
    return h


def fc_layer(previous, input_size, output_size, name):
    W = weight_variable([input_size, output_size], name)
    b = bias_variable([output_size], name)
    return tf.matmul(previous, W) + b


def autoencoder(x,y,b,dtype):
    l1 = tf.nn.tanh(fc_layer(x, 28*28, 300, "auto"))
    l2 = tf.nn.tanh(fc_layer(l1, 300, 60, "auto"))
    l3 = fc_layer(l2, 60, 30, "auto")
    l4 = tf.nn.tanh(fc_layer(l3, 30, 60, "auto"))
    l5 = tf.nn.tanh(fc_layer(l4, 60, 300, "auto"))
    out = tf.nn.relu(fc_layer(l5, 300, 28*28, "auto"))
    H = tf.constant(haarMatrix(32), name="wave", dtype=np.float32)
    
    loss = tf.reduce_mean(tf.squared_difference(x, out)) + b*tf.reduce_mean(tf.squared_difference(wavelet(x,dtype,H),wavelet(out,dtype,H)))
    mse = tf.reduce_mean(tf.squared_difference(x, out))
    return loss, mse, out, l3

def wavelet(x,dtype,H):
    T = []
    if dtype==1:
        N = batch_size
    else:
        N = test_batch_size
    paddings = tf.constant([[2, 2,], [2, 2]])
    #H = haarMatrix(32)
    #H = tf.Variable(haarMatrix(32), name="wave")
    for i in range(N):
        #T = tf.concat([T,tf.reshape(tf.matmul(tf.matmul(np.float32(H),tf.pad(tf.reshape(x[i,:],[28,28]),paddings,"CONSTANT")),np.float32(H),transpose_b=True),[-1])],0)
        T = tf.concat([T,tf.reshape(tf.matmul(tf.matmul(H,tf.pad(tf.reshape(x[i,:],[28,28]),paddings,"CONSTANT")),H,transpose_b=True),[-1])],0)
    T = tf.reshape(T,[N,1024])
    return T

loss, mse, output, latent = autoencoder(X, Y, beta, data_type)

    # and we use the Adam Optimizer for training

optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss,var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="auto"))
#optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)    
init = tf.global_variables_initializer()
#init2 = tf.initializers.variables(var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="auto"))
with tf.Session() as sess:
    # Run the initializer
    #sess.run(init)
    for t in rate:
#    if rate==0:    
        sess.run(init)
        for i in range(1, num_steps+1):            
            # Prepare Data
            # Get the next batch of MNIST data (only images are needed, not labels)
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            #batch_x, batch_y = mnist.train.next_batch(1)
    
            # Run optimization op (backprop) and cost op (to get loss value)
            _, l = sess.run([optimizer, loss], feed_dict={X: batch_x, Y: batch_y, beta: t, data_type: 1})
            # Display logs per step
            if i % display_step == 0 or i == 1:
                print('Step %i: Minibatch Loss: %f' % (i, l))
                #training_loss.append(l)
    
        n = 100
        error = []
        test_loss=[]
        for i in range(n):
            batch_x, _ = mnist.test.next_batch(test_batch_size)
            #batch_x, _ = mnist.test.next_batch(1)
            tl,l = sess.run([loss,mse], feed_dict={X: batch_x, Y: batch_y, beta: t, data_type: 0})
            error.append(l)
            test_loss.append(tl)
        
        #plt.plot(error)
        #plt.show()
        perf.append(sum(error)/n)
        perf_tl.append(sum(test_loss)/n)
    
    plt.semilogx(rate,perf)
    #plt.semilogx(rate,perf_tl)
    #plt.legend('MSE','Test loss')
    plt.xlabel('$\lambda$')
    plt.ylabel('MSE')
    plt.show