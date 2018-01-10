import numpy as np
import tensorflow as tf
import os
from scipy.io import wavfile
from text import *
from parameters import params as pm
import matplotlib.pyplot as plt
def normalize(inputs, 
              epsilon = 1e-8):
    inputs_shape = inputs.get_shape()
    params_shape = inputs_shape[-1:]
    mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
    beta= tf.Variable(tf.zeros(params_shape))
    gamma = tf.Variable(tf.ones(params_shape))
    normalized = (inputs - mean) / ( (variance + epsilon) ** (.5) )
    outputs = gamma * normalized + beta     
    return outputs

def feed_forward(inputs, w):
    outputs = [tf.matmul(inputs[i, :, :], w) for i in range(pm.batch_size)]
    outputs = tf.stack(outputs)
    return outputs

def positional_encoding(inputs,
                        num_units):

    N, T = inputs.get_shape().as_list()
    position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1])

    # First part of the PE function: sin and cos argument
    position_enc = np.array([
            [pos / np.power(10000, 2.*i/num_units) for i in range(num_units)]
            for pos in range(T)])
    # Second part, apply the cosine to even columns and sin to odds.
    position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
    position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
    # Convert to a tensor
    lookup_table = tf.convert_to_tensor(position_enc, dtype = tf.float32)
    outputs = tf.nn.embedding_lookup(lookup_table, position_ind)
    return outputs

labels = []
with open('samples/labels/labels.txt', 'r', encoding = 'utf-8') as lb:
    for l in lb.readlines():
        l = l.strip()
        l = l.split('||')[1]
        labels.append(text2label(l))
labels = np.array(labels)
audio = os.listdir('processed/wavs')
wavs = np.array([np.load(os.path.join('processed/wavs', audio[i])) for i in range(len(audio))])
x = tf.placeholder("int32", [pm.batch_size, pm.Tx], name = "x")
y = tf.placeholder("float32", [pm.batch_size, pm.Dy, pm.Ty], name = "y")
lookup_table = tf.Variable(tf.random_uniform((pm.vocab_size, pm.num_units), minval=-0.5, maxval=0.5,dtype=tf.float32), name = 'lookup_table')
x_embed = tf.nn.embedding_lookup(lookup_table, x, name = 'x_embed')
x_embed += positional_encoding(x, pm.num_units)
w1 = tf.truncated_normal((3, pm.num_units, pm.num_units), mean=0.0, stddev=0.01, dtype=tf.float32, seed=None, name='w1')
Q = feed_forward(x_embed, w1[0, :, :])
K = feed_forward(x_embed, w1[1, :, :])
V = feed_forward(x_embed, w1[2, :, :])
net = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))
net = tf.matmul(net, V)
net += x_embed
net = normalize(net)
w2 = tf.tile(tf.truncated_normal((pm.step_size, pm.num_units), mean=0.0, stddev=0.01, dtype=tf.float32, seed=None), [int(pm.num_units/pm.step_size), 1], name = 'w2')
w3 = tf.tile(tf.truncated_normal((pm.step_size, pm.num_units), mean=0.0, stddev=0.01, dtype=tf.float32, seed=None), [int(pm.num_units/pm.step_size), 1], name = 'w3')
w4 = tf.tile(tf.truncated_normal((pm.step_size, pm.num_units), mean=0.0, stddev=0.01, dtype=tf.float32, seed=None), [int(pm.num_units/pm.step_size), 1], name = 'w4')
w5 = tf.tile(tf.truncated_normal((pm.step_size, pm.num_units), mean=0.0, stddev=0.01, dtype=tf.float32, seed=None), [int(pm.num_units/pm.step_size), 1], name = 'w5')
w6 = tf.tile(tf.truncated_normal((pm.step_size, pm.num_units), mean=0.0, stddev=0.01, dtype=tf.float32, seed=None), [int(pm.num_units/pm.step_size), 1], name = 'w6')
net += feed_forward(net, w2)
net = tf.nn.relu(net)
net = normalize(net)
net += feed_forward(net, w3)
net = tf.nn.relu(net)
net = normalize(net)
net += feed_forward(net, w4)
net = tf.nn.relu(net)
net = normalize(net)
net += feed_forward(net, w5)
net = tf.nn.relu(net)
net = normalize(net)
net += feed_forward(net, w6)
net = tf.nn.relu(net)
net = normalize(net)
w7 = tf.tile(tf.truncated_normal((1, pm.Dy,), mean=0.0, stddev=0.01, dtype=tf.float32, seed=None), [pm.num_units, 1], name = 'w7')
net = feed_forward(net, w7)
net = normalize(net)
net = tf.transpose(net, [0, 2, 1])
w8 = tf.truncated_normal((pm.Tx, pm.Ty), mean=0.0, stddev=0.01, dtype=tf.float32, seed=None, name = 'w8')
yhat = feed_forward(net, w8)
loss = tf.reduce_mean(tf.abs(y - yhat), name = 'loss')
optimizer = tf.train.AdamOptimizer(learning_rate = pm.lr).minimize(loss)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(2000):
        _ = sess.run(optimizer, feed_dict = {x:labels, y:wavs})
        print('Step: ', i, 'loss: ', sess.run(loss, feed_dict = {x:labels, y:wavs}))
        if i % 100 == 0:
            ypred = sess.run(yhat, feed_dict = {x:labels, y:wavs})
            ypred = ypred[0, :, :] 
            ypred = ypred.reshape(1, -1)[0] * 10000
            ypred = ypred.astype(np.int16)
            wavfile.write('output.wav', 16000, ypred)
plt.plot(ypred)
plt.show()
