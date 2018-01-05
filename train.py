import numpy as np
import tensorflow as tf
from text import *
from parameters import params as pm
graph = tf.Graph
with graph.as_default():
    x = tf.placeholder("float32", [pm.batch_size, pm.Tx], name = "x")
    y = tf.placeholder("int32", [pm.batch_size, pm.Dy, pm.Ty], name = "y")
    
