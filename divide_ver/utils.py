#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np

np.random.seed(1337)

def batch_generator(px, vx, y, meta, corr, batch_size):
    size = px[0].shape[0]
    px_copy = px.copy()
    vx_copy = vx.copy()
    y_copy = y.copy()
    meta_copy = meta.copy()
    c_copy = corr.copy()
    i = 0
    while True:
        final_px, final_vx = [], []
        if i + batch_size <= size:
            #park data
            px_copy_c = px_copy[0][i:i + batch_size]
            px_copy_c = np.asarray(px_copy_c)
            px_copy_p = px_copy[1][i:i + batch_size]
            px_copy_p = np.asarray(px_copy_p)
            px_copy_t = px_copy[2][i:i + batch_size]
            px_copy_t = np.asarray(px_copy_t)
            final_px.extend([px_copy_c, px_copy_p, px_copy_t])

            #vd data
            vx_copy_c = vx_copy[0][i:i + batch_size]
            vx_copy_c = np.asarray(vx_copy_c)
            vx_copy_p = vx_copy[1][i:i + batch_size]
            vx_copy_p = np.asarray(vx_copy_p)
            vx_copy_t = vx_copy[2][i:i + batch_size]
            vx_copy_t = np.asarray(vx_copy_t)
            final_vx.extend([vx_copy_c, vx_copy_p, vx_copy_t])
            
            yield final_px, final_vx, y_copy[i:i + batch_size], meta_copy[i:i + batch_size], c_copy[i:i + batch_size]
            i += batch_size
        else:
            i = 0
            continue