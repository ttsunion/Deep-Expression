# -*- coding: utf-8 -*-
#/usr/bin/python3

class params:
    # signal processing
    sr = 16000 # Sampling rate.
    n_fft = 2048 # fft points (samples)
    frame_shift = 0.0125 # seconds
    frame_length = 0.05 # seconds
    hop_length = int(sr*frame_shift) # samples  This is dependent on the frame_shift.
    win_length = int(sr*frame_length) # samples This is dependent on the frame_length.
    n_mels = 80 # Number of Mel banks to generate
    sharpening_factor = 1.4 # Exponent for amplifying the predicted magnitude
    n_iter = 50 # Number of inversion iterations
    preemphasis = .97 # or None
    max_db = 100
    ref_db = 20

    # Model
    r = 4 # Reduction factor
    dropout_rate = .2
    ## Enocder
    vocab_size = 32 # [PE a-z'.?]
    embed_size = 256 # == e
    enc_layers = 7
    enc_filter_size = 5
    enc_channels = 64 # == c
    ## Decoder
    dec_layers = 4
    dec_filter_size = 5
    attention_size = 128*2 # == a
    ## Converter
    converter_layers = 5*2
    converter_filter_size = 5
    converter_channels = 256 # == v

    sinusoid = True
    attention_win_size = 3

    # data
    max_duration = 10.0 # seconds
    Tx = 20 # characters. maximum length of text.

    # training scheme
    lr = 0.001
    logdir = "logdir"
    sampledir = 'samples'
    batch_size = 4
    max_grad_norm = 100.
    max_grad_val = 5.
    num_iterations = 500000
