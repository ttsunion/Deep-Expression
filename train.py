import numpy as np
import tensorflow as tf
import os
import wave
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
def feed_forward(inputs, w):
    outputs = [tf.matmul(inputs[i, :, :], w) for i in range(pm.batch_size)]
    outputs = tf.stack(outputs)
    outputs = tf.nn.relu(outputs)
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
w2 = tf.tile(tf.truncated_normal((1, pm.num_units), mean=0.0, stddev=1.0, dtype=tf.float32, seed=None), [pm.num_units, 1], name = 'w2')
w3 = tf.tile(tf.truncated_normal((1, pm.num_units), mean=0.0, stddev=1.0, dtype=tf.float32, seed=None), [pm.num_units, 1], name = 'w3')
w4 = tf.tile(tf.truncated_normal((1, pm.num_units), mean=0.0, stddev=1.0, dtype=tf.float32, seed=None), [pm.num_units, 1], name = 'w4')
w5 = tf.tile(tf.truncated_normal((1, pm.num_units), mean=0.0, stddev=1.0, dtype=tf.float32, seed=None), [pm.num_units, 1], name = 'w5')
w6 = tf.tile(tf.truncated_normal((1, pm.num_units), mean=0.0, stddev=1.0, dtype=tf.float32, seed=None), [pm.num_units, 1], name = 'w6')
net = feed_forward(net, w2)
net = normalize(net)
net = feed_forward(net, w3)
net = normalize(net)
net = feed_forward(net, w4)
net = normalize(net)
net = feed_forward(net, w5)
net = normalize(net)
net = feed_forward(net, w6)
net = normalize(net)
w7 = tf.tile(tf.truncated_normal((1, pm.Dy,), mean=0.0, stddev=1.0, dtype=tf.float32, seed=None), [pm.num_units, 1], name = 'w7')
net = [tf.matmul(net[i, :, :], w7) for i in range(pm.batch_size)]
net = tf.stack(net)
net = tf.nn.relu(net)
net = normalize(net)
net = tf.transpose(net, [0, 2, 1])
w8 = tf.truncated_normal((pm.Tx, pm.Ty), mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name = 'w8')
net = [tf.matmul(net[i, :, :], w8) for i in range(pm.batch_size)]
yhat = tf.stack(net)
loss = tf.reduce_mean(tf.abs(y - net), name = 'loss')
optimizer = tf.train.AdamOptimizer(learning_rate = pm.lr).minimize(loss)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(2000):
        _ = sess.run(optimizer, feed_dict = {x:labels, y:wavs})
        print('loss:	', sess.run(loss, feed_dict = {x:labels, y:wavs}))
        if i % 100 == 0:
            ypred = sess.run(yhat, feed_dict = {x:labels, y:wavs}) * 2**10
            ypred = ypred[0, :, :]
            ypred = ypred.reshape(int(pm.sr * pm.max_duration), 1)
            fi = wave.open(r"test.wav", "wb")
            fi.setnchannels(1)
            fi.setsampwidth(2)
            fi.setframerate(pm.sr)
            fi.writeframes(ypred.tostring())
            fi.close()
