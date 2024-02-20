#!/usr/bin/python
# -*- coding: utf-8 -*-
from st_resnet import Graph
import tensorflow as tf
from params import Params as param
import numpy as np
import utils, data_preprocess
import pandas as pd
import datetime

np.random.seed(1337)

tf.reset_default_graph()

def label_process(park_y, vd_y):
    data_temp = []
    for i in range(len(park_y)):
        # 疊深度
        #temp = [park_y[i], vd_y[i]] #停車場在前，VD在後
        temp = [vd_y[i], park_y[i]]  #VD在前，停車場在後
        temp = data_preprocess.revise_data_shape(temp) #疊資料，shape:(2, w, h) -> shape:(w, h, 2)
        data_temp.append(temp)
        
        """
        # 疊長度
        p_temp, v_temp = [], []
        for n in range(len(park_y[i])):
            row = []
            for j in range(len(park_y[i][n])):
                row.append([park_y[i][n][j]])
            p_temp.append(row)
        for n in range(len(vd_y[i])):
            row = []
            for j in range(len(vd_y[i][n])):
                row.append([vd_y[i][n][j]])
            v_temp.append(row)

        temp = np.concatenate([p_temp, v_temp], axis=1)
        data_temp.append(temp)
        """
    return data_temp

start_time = datetime.datetime.now() #code start time
#讀取數據
park_info, vd_info, corr_info = data_preprocess.load_data(len_closeness=3, len_period=1, len_trend=1) #設定closeness、period、trend長度取資料
y_train = label_process(park_info[1], vd_info[1]) #park、vd的label擺一起
y_test = label_process(park_info[3], vd_info[3]) #park、vd的label擺一起

#park train data
p_x_train_c = np.array(park_info[0][0])
p_x_train_p = np.array(park_info[0][1])
p_x_train_t = np.array(park_info[0][2])

#park meta data，meta data都一樣，所以看park的就可以
meta_train = np.array(park_info[7])
meta_test = np.array(park_info[8])

#park validataion data
p_x_test_c = np.array(park_info[2][0])
p_x_test_p = np.array(park_info[2][1])
p_x_test_t = np.array(park_info[2][2])

p_x_train = [p_x_train_c, p_x_train_p, p_x_train_t]
p_x_test = [p_x_test_c, p_x_test_p, p_x_test_t]

#vd train data
v_x_train_c = np.array(vd_info[0][0])
v_x_train_p = np.array(vd_info[0][1])
v_x_train_t = np.array(vd_info[0][2])

#vd validataion data
v_x_test_c = np.array(vd_info[2][0])
v_x_test_p = np.array(vd_info[2][1])
v_x_test_t = np.array(vd_info[2][2])

v_x_train = [v_x_train_c, v_x_train_p, v_x_train_t]
v_x_test = [v_x_test_c, v_x_test_p, v_x_test_t]

#將資料做batch
train_batch_generator = utils.batch_generator(p_x_train, v_x_train, y_train, meta_train, corr_info[0], param.batch_size)
test_batch_generator = utils.batch_generator(p_x_test, v_x_test, y_test, meta_test, corr_info[1], param.batch_size)

print('Start learning...')
g = Graph()
with tf.device(':/gpu:0'):
    config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        
        for epoch in range(param.num_epochs):
            loss_train = 0
            loss_val = 0

            #train
            num_batches = p_x_train[0].shape[0] // param.batch_size
            for b in range(num_batches):
                p_x_batch, v_x_batch, y_batch, meta_batch, c_batch = next(train_batch_generator) #分批將資料輸入到模型
                #for park, type改為list
                p_x_closeness = np.array(p_x_batch[0].tolist())
                p_x_period = np.array(p_x_batch[1].tolist())
                p_x_trend = np.array(p_x_batch[2].tolist())

                #for vd, type改為list
                v_x_closeness = np.array(v_x_batch[0].tolist())
                v_x_period = np.array(v_x_batch[1].tolist())
                v_x_trend = np.array(v_x_batch[2].tolist())
                
                loss_tr, _, summary = sess.run([g.loss, g.optimizer, g.merged], feed_dict={g.p_c_inp:p_x_closeness, g.p_p_inp:p_x_period, g.p_t_inp:p_x_trend, g.v_c_inp:v_x_closeness, g.v_p_inp:v_x_period, g.v_t_inp:v_x_trend,
                                                                                            g.output:y_batch, g.external:meta_batch, g.correlation:c_batch})
                loss_train = loss_tr * param.delta + loss_train * (1 - param.delta)

            #validataion
            num_batches = p_x_test[0].shape[0] // param.batch_size
            for b in range(num_batches):
                p_x_batch, v_x_batch, y_batch, meta_batch, c_batch = next(test_batch_generator)
                #for park
                p_x_closeness = np.array(p_x_batch[0].tolist())
                p_x_period = np.array(p_x_batch[1].tolist())
                p_x_trend = np.array(p_x_batch[2].tolist())

                #for vd
                v_x_closeness = np.array(v_x_batch[0].tolist())
                v_x_period = np.array(v_x_batch[1].tolist())
                v_x_trend = np.array(v_x_batch[2].tolist())

                pre, g_truth, loss_v, summary = sess.run([g.x_res, g.output, g.loss, g.merged], feed_dict={g.p_c_inp:p_x_closeness, g.p_p_inp:p_x_period, g.p_t_inp:p_x_trend, g.v_c_inp:v_x_closeness, g.v_p_inp:v_x_period, g.v_t_inp:v_x_trend,
                                                                            g.output:y_batch, g.external:meta_batch, g.correlation:c_batch})
                loss_val += loss_v


            if num_batches != 0:
                loss_val /= num_batches
                
            print('Epoch: {} - loss: {:.6f} - val_loss: {:.6f}'.format(epoch, loss_train, loss_val))

            if epoch % 10 == 0:
                g.saver.save(sess, 'D:/YZU/Master/paper/old_paper/Xinyi_park_vd/divide_ver/New_Model/save_net_1.ckpt', epoch)


end_time = datetime.datetime.now() #code end time
print('start time: {0}:{1}'.format(start_time.hour, start_time.minute))
print('end time: {0}:{1}'.format(end_time.hour, end_time.minute))
if end_time.hour == start_time.hour:
    print('Program spent {0} mins'.format(end_time.minute - start_time.minute))
else:
    print('Program spent {0} mins'.format((60 + int(end_time.minute)) - int(start_time.minute)))
print('Finished Training.')