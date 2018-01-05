import numpy as np
import tensorflow as tf
import os
from text import *
from parameters import params as pm

graph = tf.Graph
labels = []
with open('samples/labels/labels.txt', 'r', encoding = 'utf-8') as lb:
    for l in lb.readlines():
        l = l.strip()
        l = l.split('||')[1]
        labels.append(text2label(l))
labels = np.array(labels)
with graph.as_default():
    x = tf.placeholder("float32", [pm.batch_size, pm.Tx], name = "x")
    y = tf.placeholder("int32", [pm.batch_size, pm.Dy, pm.Ty], name = "y")
    
