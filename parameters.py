# -*- coding: utf-8 -*-
# #/usr/bin/python3

class params:
    # signal processing
    sr = 16000 # Sampling rate.

    # data
    max_duration = 10.0 # seconds
    Tx = 20 # characters. maximum length of text.

    # training scheme
    lr = 0.001
    logdir = "logdir"
