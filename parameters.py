# -*- coding: utf-8 -*-
# #/usr/bin/python3

class params:
    # signal processing
    sr = 16000 # Sampling rate.

    # data
    max_duration = 10.0 # seconds
    Tx = 20 # characters. maximum length of text.
    vocab_size = 48 # unique chars in dictionary
    frame_length = 0.05 # seconds
    Ty = int(sr * frame_length) # signals. maximum length  of signal.
    Dy = int(sr * max_duration/Ty)
    num_filters = 2048
    step_size = 4
    # training scheme
    lr = 0.005
    batch_size = 4
    num_units = 512
    logdir = "logdir"
