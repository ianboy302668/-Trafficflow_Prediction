#!/usr/bin/python
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from params import Params as param
import modules as my

class Graph(object):
    def __init__(self):
        tf.reset_default_graph() #測試時用
        self.graph = tf.Graph()

        # self.state = True         #訓練時用
        self.state = False       #測試時用

        B, H, W, C, P, T, O, F, U = param.batch_size, param.map_height, param.map_width, param.closeness_sequence_length*param.nb_flow, param.period_sequence_length*param.nb_flow, param.trend_sequence_length*param.nb_flow, param.num_of_output ,param.num_of_filters, param.num_of_residual_units
        ##-----------------------------------------訓練用參數---------------------------------------##
        #park
        self.p_c_inp = tf.placeholder(tf.float32, shape=[B, H, W, C], name='park_closeness')
        self.p_p_inp = tf.placeholder(tf.float32, shape=[B, H, W, P], name='park_period')
        self.p_t_inp = tf.placeholder(tf.float32, shape=[B, H, W, T], name='park_trend')
        #vd
        self.v_c_inp = tf.placeholder(tf.float32, shape=[B, H, W, C], name='vd_closeness')
        self.v_p_inp = tf.placeholder(tf.float32, shape=[B, H, W, P], name='vd_period')
        self.v_t_inp = tf.placeholder(tf.float32, shape=[B, H, W, T], name='vd_trend')
        self.output = tf.placeholder(tf.float32, shape=[B, H, W, O], name='output')

        self.external = tf.placeholder(tf.float32, shape=[B, 8], name='external')
        self.correlation = tf.placeholder(tf.float32, shape=[B, 27, 10], name='correlation')
        ##---------------------------------------------------------------------------------------##

        ##-----------------------------------------測試用參數--------------------------------------##
        #park
        self.p_c_inp_t = tf.placeholder(tf.float32, shape=[1, H, W, C], name='park_closeness')
        self.p_p_inp_t = tf.placeholder(tf.float32, shape=[1, H, W, P], name='park_period')
        self.p_t_inp_t = tf.placeholder(tf.float32, shape=[1, H, W, T], name='park_trend')
        #vd
        self.v_c_inp_t = tf.placeholder(tf.float32, shape=[1, H, W, C], name='vd_closeness')
        self.v_p_inp_t = tf.placeholder(tf.float32, shape=[1, H, W, P], name='vd_period')
        self.v_t_inp_t = tf.placeholder(tf.float32, shape=[1, H, W, T], name='vd_trend')

        self.external_t = tf.placeholder(tf.float32, shape=[1, 8], name='external')
        self.correlation_t = tf.placeholder(tf.float32, shape=[1, 27, 10], name='correlation')
        ##---------------------------------------------------------------------------------------##

        if self.state == True: #訓練用
            self.pc_ = self.p_c_inp
            self.pp_ = self.p_p_inp
            self.pt_ = self.p_t_inp

            self.vc_ = self.v_c_inp
            self.vp_ = self.v_p_inp
            self.vt_ = self.v_t_inp
            self.external_ = self.external
            self.correlation_ = self.correlation
        else: #測試用
            self.pc_ = self.p_c_inp_t
            self.pp_ = self.p_p_inp_t
            self.pt_ = self.p_t_inp_t

            self.vc_ = self.v_c_inp_t
            self.vp_ = self.v_p_inp_t
            self.vt_ = self.v_t_inp_t
            self.external_ = self.external_t
            self.correlation_ = self.correlation_t
        
        #for park
        # module 1: capturing closeness (recent)
        self.p_closeness_output = my.ResInput(inputs=self.pc_, filters=F, kernel_size=(3, 3), scope='park_closeness_input', reuse=None)
        self.p_closeness_output = my.ResNet(inputs=self.p_closeness_output, filters=F, kernel_size=(3, 3), repeats=U, scope='park_resnet', reuse=None)
        self.p_closeness_output = my.ResOutput(inputs=self.p_closeness_output, filters=1, kernel_size=(3, 3), scope='park_resnet_output', reuse=None)
        
        # module 2: capturing period (near)
        self.p_period_output = my.ResInput(inputs=self.pp_, filters=F, kernel_size=(3, 3), scope='park_period_input', reuse=None)
        self.p_period_output = my.ResNet(inputs=self.p_period_output, filters=F, kernel_size=(3, 3), repeats=U, scope='park_resnet', reuse=True)
        self.p_period_output = my.ResOutput(inputs=self.p_period_output, filters=1, kernel_size=(3, 3), scope='park_resnet_output', reuse=True)
        
        # module 3: capturing trend (distant)
        self.p_trend_output = my.ResInput(inputs=self.pt_, filters=F, kernel_size=(3, 3), scope='park_trend_input', reuse=None)
        self.p_trend_output = my.ResNet(inputs=self.p_trend_output, filters=F, kernel_size=(3, 3), repeats=U, scope='park_resnet', reuse=True)
        self.p_trend_output = my.ResOutput(inputs=self.p_trend_output, filters=1, kernel_size=(3, 3), scope='park_resnet_output', reuse=True)
        
        # module 4: external component
        self.external_output = my.External(inputs=self.external_, units=10)

        self.p_res = my.Fusion(self.p_closeness_output, self.p_period_output, self.p_trend_output, self.external_output, scope='park_fusion', shape=[W, W])
        
        #for vd
        # module 1: capturing closeness (recent)
        self.v_closeness_output = my.ResInput(inputs=self.vc_, filters=F, kernel_size=(3, 3), scope='vd_closeness_input', reuse=None)
        self.v_closeness_output = my.ResNet(inputs=self.v_closeness_output, filters=F, kernel_size=(3, 3), repeats=U, scope='vd_resnet', reuse=None)
        self.v_closeness_output = my.ResOutput(inputs=self.v_closeness_output, filters=1, kernel_size=(3, 3), scope='vd_resnet_output', reuse=None)
        
        # module 2: capturing period (near)
        self.v_period_output = my.ResInput(inputs=self.vp_, filters=F, kernel_size=(3, 3), scope='vd_period_input', reuse=None)
        self.v_period_output = my.ResNet(inputs=self.v_period_output, filters=F, kernel_size=(3, 3), repeats=U, scope='vd_resnet', reuse=True)
        self.v_period_output = my.ResOutput(inputs=self.v_period_output, filters=1, kernel_size=(3, 3), scope='vd_resnet_output', reuse=True)
        
        # module 3: capturing trend (distant)
        self.v_trend_output = my.ResInput(inputs=self.vt_, filters=F, kernel_size=(3, 3), scope='vd_trend_input', reuse=None)
        self.v_trend_output = my.ResNet(inputs=self.v_trend_output, filters=F, kernel_size=(3, 3), repeats=U, scope='vd_resnet', reuse=True)
        self.v_trend_output = my.ResOutput(inputs=self.v_trend_output, filters=1, kernel_size=(3, 3), scope='vd_resnet_output', reuse=True)
        
        # module 4: external component
        self.external_output = my.External(inputs=self.external_, units=10)

        self.v_res = my.Fusion(self.v_closeness_output, self.v_period_output, self.v_trend_output, self.external_output, scope='vd_fusion', shape=[W, W])
        
        #合併兩個stream的輸出
        #self.x_res = my.Merge_Stream(self.p_res, self.v_res, self.correlation_) #加關聯度
        self.x_res = my.Merge_Res(self.p_res, self.v_res)                        #沒加關聯度
        
        if self.state == True:
            self.loss = tf.sqrt(tf.reduce_sum(tf.pow(self.x_res - self.output, 2)) / tf.cast((self.x_res.shape[0]), tf.float32))
            self.optimizer = tf.train.AdamOptimizer(learning_rate=param.lr).minimize(self.loss)
            
            tf.summary.scalar('loss', self.loss)
        self.merged = tf.summary.merge_all()
        self.saver = tf.train.Saver(max_to_keep=None)