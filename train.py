import numpy as np
import tensorflow as tf
import os
from text import *
from parameters import params as pm

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
w1 = tf.truncated_normal((3, pm.num_units, pm.num_units), mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name='w1')
Q = tf.nn.relu(tf.scan(lambda a, x: tf.matmul(x, w1[0, :, :]), x_embed), name = 'Q')
K = tf.nn.relu(tf.scan(lambda a, x: tf.matmul(x, w1[1, :, :]), x_embed), name = 'K')
V = tf.nn.relu(tf.scan(lambda a, x: tf.matmul(x, w1[2, :, :]), x_embed), name = 'V')
net = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))
net = tf.matmul(net, V)
net += x_embed
net = normalize(net)
w2 = tf.tile(tf.truncated_normal((1, pm.Dy * pm.num_units), mean=0.0, stddev=1.0, dtype=tf.float32, seed=None), [pm.Tx * pm.num_units, 1], name = 'w2')
net = tf.reshape(net, [-1, pm.Tx * pm.num_units])
net = tf.matmul(net, w2)
net = tf.reshape(net, [-1, pm.Dy, pm.num_units])
net = tf.nn.relu(net)
net = normalize(net)
net = tf.reshape(net, [-1, pm.Dy * pm.num_units])
w3 = tf.truncated_normal((pm.Dy * pm.num_units, int(pm.sr * pm.max_duration)), mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name = 'w3')
net = tf.matmul(net, w3)
net = tf.reshape(net, [-1, pm.Dy, pm.Ty])
loss = tf.reduce_mean(tf.abs(y - net), name = 'loss')
optimizer = tf.train.AdamOptimizer(learning_rate = pm.lr).minimize(loss)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1000):
        result = sess.run(optimizer, feed_dict = {x:labels, y:wavs})
        print('loss:	', sess.run(loss, feed_dict = {x:labels, y:wavs}))
