#!/usr/bin/python
# -*- coding: utf-8 -*-
class Params(object):
    batch_size = 32
    map_height = 15
    map_width = 16
    closeness_sequence_length = 3
    period_sequence_length = 1
    trend_sequence_length = 1
    nb_flow = 1
    num_of_filters = 64
    num_of_residual_units = 4
    num_of_output = 2
    delta = 0.5
    epsilon = 1e-7
    beta1 = 0.8
    beta2 = 0.999
    lr = 0.0002
    num_epochs = 101