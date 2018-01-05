# -*- coding: utf-8 -*-
# #/usr/bin/python3

class params:
    # signal processing
    sr = 16000 # Sampling rate.

    # data
    max_duration = 10.0 # seconds
    Tx = 20 # characters. maximum length of text.
    frame_length = 0.05 # seconds
    Ty = int(sr * frame_length) # signals. maximum length  of signal.
    Dy = int(sr * max_duration/Ty)
    # training scheme
    lr = 0.001
    batch_size = 4
    logdir = "logdir"
