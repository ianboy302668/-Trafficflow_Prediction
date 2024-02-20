#!/usr/bin/python
# -*- coding: utf-8 -*-
import tensorflow as tf
from st_resnet import Graph
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time, math, os, csv
import mape_calcu, check

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.reset_default_graph()

#字串轉時間格式
def string2timestamp(strings, T=96):
    strings = [strings]
    for t in strings: #strings裡面為20180301-00:00形式
        year, month, day, hour, tm_min = int(t[:4]), int(t[4:6]), int(t[6:8]), int(t[8:10]), int(t[10:])
        timestamps = pd.Timestamp(datetime(year, month, day, hour, tm_min)) #轉成2018-03-01 00:00形式
        
    return timestamps

#時間格式轉字串
def timestamp2string(timestamps, T=96):
    timestamp = []
    for ts in timestamps:
        year = ts.date().strftime('%Y')
        month = ts.date().strftime('%m')
        day = ts.date().strftime('%d')
        tm_hour = int(ts.to_pydatetime().hour)
        tm_min = int(ts.to_pydatetime().minute)
        if tm_hour < 10:
            tm_hour = '0' + str(tm_hour)
        if tm_min < 10:
            tm_min = '0' + str(tm_min)
        Date = year + month + day + str(tm_hour) + str(tm_min) + '\n'
        timestamp.append(Date)

    return timestamp

#分別對資料做min-max normalization
def MinMaxNormalization(X, state=None):
    if state == 'park':
        x_max = 2000
    else:
        x_max = 120
    x_min = 0

    X = 1. * (X - x_min) / (x_max - x_min)

    return X

#將預測出來的值反算回去
def reverse_MinMaxNormalization(X):
    X = np.array(X[0][0])
    for i in range(len(X)):
        for j in range(len(X[i])):
            X[i][j][0] = 120 * X[i][j][0]
            X[i][j][1] = 2000 * X[i][j][1]

    return X

#疊資料，shape:(2, w, h) -> shape:(w, h, 2)
def revise_data_shape(x, state=None):
    if state == True:
        x1, x2 = x[0], x[1]
        column = []
        for i in range(len(x1)):
            column.append([x1[i], x2[i]])
        x_final = column.copy()
    else:
        x1, x2 = x[0], x[1]
        column = []
        for i in range(len(x1)):
            row = []
            for j in range(len(x1[i])):
                row.append([x1[i][j], x2[i][j]])
            column.append(row)
        x_final = column.copy()
    
    return x_final

#把資料resize到地圖的大小，因為有時候一個grid有兩個站點，在csv中會分別顯示資料
#調整grid
def checkvalue(x, latlon, name, state):
    mapdata, namedata = [], []
    for i in range(len(x)):
        Index = True
        if i >= 2:
            #跳過同網格內的點
            if latlon[i][0] == latlon[i-1][0] and latlon[i][1] == latlon[i-1][1]:
                Index = False
        if Index:
            # print('index:{0}'.format(latlon[i]))
            #檢查同一個網格內是否有多個停車場
            if i != (len(x) - 1):
                #一樣
                if latlon[i][0] == latlon[i+1][0] and latlon[i][1] == latlon[i+1][1]:
                    if state == True:
                        s1, s2, num = 0, 0, 0
                        #如果有，就檢查這幾個停車場是否為私營
                        for j in range(2):
                            if j == 0:
                                if str(name[i][0]) == 'J' or str(name[i][0]) == 'B':
                                    num += 1
                                    s1 = 1 #代表為私營
                            elif j == 1:
                                if str(name[i+1][0]) == 'J' or str(name[i+1][0]) == 'B':
                                    num += 1
                                    s2 = 1 #代表為私營
                        #num=2代表兩個都是私營的，只需要存其中一個資料
                        if num == 2:
                            x[i] = 0
                            mapdata.append(x[i])
                            namedata.append(name[i])
                        elif num == 1:
                            if s1 == 1:
                                mapdata.append(x[i+1])
                                namedata.append(name[i+1])
                            elif s2 == 1:
                                mapdata.append(x[i])
                                namedata.append(name[i])
                        elif num == 0:
                            temp = x[i] + x[i+1]
                            mapdata.append(temp)
                            namedata.append(name[i])
                    else:
                        temp = math.ceil((x[i] + x[i+1])/2)
                        x[i] = temp
                        mapdata.append(x[i])
                        namedata.append(name[i])
                #不一樣
                else:
                    if str(x[i]) == 'nan':
                        x[i] = 0
                        mapdata.append(x[i])
                        namedata.append(name[i])
                    else:
                        mapdata.append(x[i])
                        namedata.append(name[i])
            elif i == (len(x) - 1):
                if str(x[i]) == 'nan':
                    x[i] = 0
                    mapdata.append(x[i])
                    namedata.append(name[i])
                else:
                    mapdata.append(x[i])
                    namedata.append(name[i])

    return mapdata

def get_label(datedata):
    state_data, labels = [], []
    only_park, only_vd = [], []

    #取停車場的ground truth
    park_path = 'D:/YZU/Master/paper/old_paper/Xinyi_park_vd/park/all_history_data/' + str(datedata[:6])
    park_file_path = park_path + '/new_park_history_' + datedata[:8] + '.csv'
    datatime = datedata[8:10] + ':' + str(datedata[10:12])

    park_data = pd.read_csv(park_file_path, encoding='ISO-8859-1')
    park_time_title = park_data.columns[4:]
    park_latlon = park_data[['Latitude', 'Longitude']].values
    park_id = park_data['id'].values

    #取VD的ground truth
    vd_path = 'D:/YZU/Master/paper/old_paper/Xinyi_park_vd/vd/all_history_data/' + str(datedata[:6])
    datedata_time = datedata[:4] + '-' + datedata[4:6] + '-' + datedata[6:8]
    vd_file_path = vd_path + '/信義區VD歷史資料_change_' + datedata_time + '.csv'

    vd_data = pd.read_csv(vd_file_path, encoding='ISO-8859-1')
    vd_time_title = vd_data.columns[5:101]
    vd_latlon = vd_data[['Latitude', 'Longitude']].values
    vd_id = vd_data['VD_id'].values

    for p_tt, v_tt in zip(park_time_title, vd_time_title):
        if p_tt.find(str(datatime)) != -1:
            #for park data
            Index = list(park_time_title).index(p_tt)
            parkdata = list(park_data[park_time_title[Index]].values)
            parkdata = checkvalue(parkdata, park_latlon, park_id, state=True)
            state_data.append(parkdata)
        if v_tt.find(str(datatime)) != -1:
            #for vd data
            Index = list(vd_time_title).index(v_tt)
            vddata = list(vd_data[vd_time_title[Index]].values)
            vddata = checkvalue(vddata, vd_latlon, vd_id, state=False)
            state_data.append(vddata)

    #我們所需的停車場站點index的csv表
    park_index_file = 'D:/YZU/Master/paper/old_paper/Xinyi_park_vd/park/park_index.csv'
    park_test = pd.read_csv(park_index_file)
    park_state_title = park_test.columns[1:]

    #我們所需的VD站點index的csv表
    vd_index_file = 'D:/YZU/Master/paper/old_paper/Xinyi_park_vd/vd/vd_index.csv'
    vd_test = pd.read_csv(vd_index_file)
    vd_state_title = vd_test.columns[1:]

    #把那些站點的資料挑出來，另外用list存
    for i in range(len(park_state_title)):
        only_park.append(state_data[0][int(park_state_title[i])])
    for j in range(len(vd_state_title)):
        only_vd.append(state_data[1][int(vd_state_title[j])])

    #data_temp = [state_data[0], state_data[1]]
    data_temp = [state_data[1], state_data[0]]
    data_temp = revise_data_shape(data_temp, state=True)
    labels.append(data_temp)

    return labels, only_park, only_vd

#把同時間的歷史資料疊在一起
def revise_shape(x):
    if len(x) == 3:
        x1, x2, x3 = x[0], x[1], x[2]
        column = []
        for i in range(len(x2)):
            row = []
            for j in range(len(x2[i])):
                row.append([x1[i][j], x2[i][j], x3[i][j]])
            column.append(row)
        x_final = column.copy()
    elif len(x) == 4:
        x1, x2, x3, x4 = x[0], x[1], x[2], x[3]
        column = []
        for i in range(len(x2)):
            row = []
            for j in range(len(x2[i])):
                row.append([x1[i][j], x2[i][j], x3[i][j], x4[i][j]])
            column.append(row)
        x_final = column.copy()
    elif len(x) == 1:
        x1 = x[0]
        column = []
        for i in range(len(x1)):
            row = []
            for j in range(len(x1[i])):
                row.append([x1[i][j]])
            column.append(row)
        x_final = column.copy()
    else:
        x1 = x.copy()
        column = []
        for i in range(len(x1)):
            row = []
            for j in range(len(x1[i])):
                row.append([x1[i][j]])
            column.append(row)
        x_final = column.copy()

    return x_final

#找出所需的時間點
def create_timestamp(timestamp, len_closeness=3, len_period=1, len_trend=1):
    offset_frame = pd.DateOffset(minutes=24 * 60 // 96)
    depends = [list(range(1, len_closeness+1)), [1 * 96 * j for j in range(1, len_period+1)], [7 * 96 * j for j in range(1, len_trend+1)]]
    xc_time = [timestamp - j * offset_frame for j in depends[0]]
    xp_time = [timestamp - j * offset_frame for j in depends[1]]
    xt_time = [timestamp - j * offset_frame for j in depends[2]]
    
    xc_time = timestamp2string(xc_time)
    xp_time = timestamp2string(xp_time)
    xt_time = timestamp2string(xt_time)

    return xc_time, xp_time, xt_time

#external factor轉one-hot encoding
def timestamp2vec(timestamps):
    timestamps = [timestamps]
    vec = [time.strptime(str(t.date().strftime('%Y') + t.date().strftime('%m') + t.date().strftime('%d')), '%Y%m%d').tm_wday for t in timestamps]
    ret = []
    for i in vec:
        v = [0 for _ in range(7)]
        v[i] = 1 #對應的星期標記為1
        if i >= 5:
            v.append(0) #假日
        else:
            v.append(1) #平日
        ret.append(v)

    return np.asarray(ret)

def read_data(datedata, timestamp, len_closeness=3, len_period=1, len_trend=1):
    park_xc, park_xp, park_xt = [], [], []
    vd_xc, vd_xp, vd_xt = [], [], []

    #返回需要的日期-時間點
    xc_time, xp_time, xt_time = create_timestamp(timestamp)
    print(f'xc_time: {xc_time}, xp_time: {xp_time}, xt_time: {xt_time}')
    #讀取c, p, t的時間點資料
    #closeness
    for xc_t in xc_time:
        park_path = 'D:/YZU/Master/paper/old_paper/Xinyi_park_vd/park/new_history/' + str(xc_t[:6])
        park_predict_date = xc_t[:8]
        park_predict_time = str(xc_t[8:10]) + ':' + str(xc_t[10:12])

        vd_path = 'D:/YZU/Master/paper/old_paper/Xinyi_park_vd/vd/new_history/' + str(xc_t[:6])
        vd_predict_date = xc_t[:4] + '-' + xc_t[4:6] + '-' + xc_t[6:8]
        vd_predict_time = str(xc_t[8:10]) + ':' + str(xc_t[10:12])

        #park data
        park_file_path = park_path + str('/new_park_history_') + str(park_predict_date) + '.csv'
        park_c = pd.read_csv(park_file_path, encoding='ISO-8859-1')
        park_time_title = park_c.columns[4:]
        park_latlon = park_c[['Latitude', 'Longitude']].values
        park_id = park_c['id'].values
        
        #vd data
        vd_file_path = vd_path + str('/信義區VD歷史資料_change_') + str(vd_predict_date) + '.csv'
        vd_c = pd.read_csv(vd_file_path, encoding='ISO-8859-1')
        vd_time_title = vd_c.columns[5:101]
        vd_latlon = vd_c[['Latitude', 'Longitude']].values
        vd_id = vd_c['VD_id'].values

        print(f'park_time size: {len(park_time_title)}') # debug
        print(f'vd_time size: {len(vd_time_title)}') # debug
        park_data, vd_data = [], []
        for p_t, v_t in zip(park_time_title, vd_time_title):
            if p_t.find(str(park_predict_time)) != -1:
                park_index = list(park_time_title).index(p_t)
                p_d = list(park_c[park_time_title[park_index]].values)
                p_d = checkvalue(p_d, park_latlon, park_id, state=True)
                print(f'p_d size: {len(p_d)}') # debug
                # park_data = np.reshape(p_d, (15, 16))  # debug
                # print(park_data)  # debug
                p_d = np.reshape(p_d, (15, 16))
                park_data.append(p_d)
            if v_t.find(str(vd_predict_time)) != -1:
                vd_index = list(vd_time_title).index(v_t)
                v_d = list(vd_c[vd_time_title[vd_index]].values)
                v_d = checkvalue(v_d, vd_latlon, vd_id, state=False)
                print(f'v_d size: {len(v_d)}') # debug
                v_d = np.reshape(v_d, (15, 16))
                vd_data.append(v_d)
        
        # print(f'park_data size: {len(park_data)}\n') # debug
        # print(park_data)# debug
        # park_data = np.reshape(park_data, (15, 16))
        # vd_data = np.reshape(vd_data, (15, 16))
        park_data = np.asarray(park_data)
        vd_data = np.asarray(vd_data)
        park_data = MinMaxNormalization(park_data, state='park')
        vd_data = MinMaxNormalization(vd_data, state='vd')
        park_xc.append(park_data)
        vd_xc.append(vd_data)

    #period
    for xp_t in xp_time:
        park_path = 'D:/YZU/Master/paper/old_paper/Xinyi_park_vd/park/all_history_data/' + str(xp_t[:6])
        park_predict_date = xp_t[:8]
        park_predict_time = str(xp_t[8:10]) + ':' + str(xp_t[10:12])

        vd_path = 'D:/YZU/Master/paper/old_paper/Xinyi_park_vd/vd/all_history_data/' + str(xp_t[:6])
        vd_predict_date = xp_t[:4] + '-' + xp_t[4:6] + '-' + xp_t[6:8]
        vd_predict_time = str(xp_t[8:10]) + ':' + str(xp_t[10:12])

        #park data
        park_file_path = park_path + str('/new_park_history_') + str(park_predict_date) + '.csv'
        park_p = pd.read_csv(park_file_path, encoding='ISO-8859-1')
        park_time_title = park_p.columns[4:]
        park_latlon = park_p[['Latitude', 'Longitude']].values
        park_id = park_p['id'].values

        #vd data
        vd_file_path = vd_path + str('/信義區VD歷史資料_change_') + str(vd_predict_date) + '.csv'
        vd_p = pd.read_csv(vd_file_path, encoding='ISO-8859-1')
        vd_time_title = vd_p.columns[5:101]
        vd_latlon = vd_p[['Latitude', 'Longitude']].values
        vd_id = vd_p['VD_id'].values

        park_data, vd_data = [], []
        for p_t, v_t in zip(park_time_title, vd_time_title):
            if p_t.find(str(park_predict_time)) != -1:
                park_index = list(park_time_title).index(p_t)
                p_d = list(park_p[park_time_title[park_index]].values)
                p_d = checkvalue(p_d, park_latlon, park_id, state=True)
                print(f'[period]p_d size: {len(p_d)}') # debug
                # p_d = np.reshape(p_d, (15, 16))
                park_data.append(p_d)
            if v_t.find(str(vd_predict_time)) != -1:
                vd_index = list(vd_time_title).index(v_t)
                v_d = list(vd_p[vd_time_title[vd_index]].values)
                v_d = checkvalue(v_d, vd_latlon, vd_id, state=False)
                print(f'[period]v_d size: {len(v_d)}') # debug
                # v_d = np.reshape(v_d, (15, 16))
                vd_data.append(v_d)
        
        park_data = np.reshape(park_data, (15, 16))
        vd_data = np.reshape(vd_data, (15, 16))
        park_data = np.asarray(park_data)
        vd_data = np.asarray(vd_data)
        park_data = MinMaxNormalization(park_data, state='park')
        vd_data = MinMaxNormalization(vd_data, state='vd')
        park_xp.append(park_data)
        vd_xp.append(vd_data)

    #trend
    for xt_t in xt_time:
        park_path = 'D:/YZU/Master/paper/old_paper/Xinyi_park_vd/park/all_history_data/' + str(xt_t[:6])
        park_predict_date = xt_t[:8]
        park_predict_time = str(xt_t[8:10]) + ':' + str(xt_t[10:12])

        vd_path = 'D:/YZU/Master/paper/old_paper/Xinyi_park_vd/vd/all_history_data/' + str(xt_t[:6])
        vd_predict_date = xt_t[:4] + '-' + xt_t[4:6] + '-' + xt_t[6:8]
        vd_predict_time = str(xt_t[8:10]) + ':' + str(xt_t[10:12])

        #park data
        park_file_path = park_path + str('/new_park_history_') + str(park_predict_date) + '.csv'
        park_t = pd.read_csv(park_file_path, encoding='ISO-8859-1')
        park_time_title = park_t.columns[4:]
        park_latlon = park_t[['Latitude', 'Longitude']].values
        park_id = park_t['id'].values

        #vd data
        vd_file_path = vd_path + str('/信義區VD歷史資料_change_') + str(vd_predict_date) + '.csv'
        vd_t = pd.read_csv(vd_file_path, encoding='ISO-8859-1')
        vd_time_title = vd_t.columns[5:101]
        vd_latlon = vd_t[['Latitude', 'Longitude']].values
        vd_id = vd_t['VD_id'].values

        park_data, vd_data = [], []
        for p_t, v_t in zip(park_time_title, vd_time_title):
            if p_t.find(str(park_predict_time)) != -1:
                park_index = list(park_time_title).index(p_t)
                p_d = list(park_t[park_time_title[park_index]].values)
                p_d = checkvalue(p_d, park_latlon, park_id, state=True)
                park_data.append(p_d)
            if v_t.find(str(vd_predict_time)) != -1:
                vd_index = list(vd_time_title).index(v_t)
                v_d = list(vd_t[vd_time_title[vd_index]].values)
                v_d = checkvalue(v_d, vd_latlon, vd_id, state=False)
                vd_data.append(v_d)

        park_data = np.reshape(park_data, (15, 16))
        vd_data = np.reshape(vd_data, (15, 16))
        park_data = np.asarray(park_data)
        vd_data = np.asarray(vd_data)
        park_data = MinMaxNormalization(park_data, state='park')
        vd_data = MinMaxNormalization(vd_data, state='vd')
        park_xt.append(park_data)
        vd_xt.append(vd_data)

    final_park_xc = revise_shape(park_xc)
    final_park_xp = revise_shape(park_xp)
    final_park_xt = revise_shape(park_xt)
    final_vd_xc = revise_shape(vd_xc)
    final_vd_xp = revise_shape(vd_xp)
    final_vd_xt = revise_shape(vd_xt)

    park_XC, park_XP, park_XT = [], [], []
    vd_XC, vd_XP, vd_XT = [], [], []
    if len_closeness > 0:
        park_XC.append(final_park_xc)
        vd_XC.append(final_vd_xc)
    if len_period > 0:
        park_XP.append(final_park_xp)
        vd_XP.append(final_vd_xp)
    if len_trend > 0:
        park_XT.append(final_park_xt)
        vd_XT.append(final_vd_xt)

    meta_feature = timestamp2vec(timestamp)
    park_info = [park_XC, park_XP, park_XT]
    vd_info = [vd_XC, vd_XP, vd_XT]
    
    #correlation data
    corr_path = 'D:/YZU/Master/paper/old_paper/Xinyi_park_vd/spearman_corr_.csv'
    corr_csv = pd.read_csv(corr_path, encoding='ISO-8859-1')
    cf = []
    for i in range(27):
        corr_data = list(corr_csv.loc[i][1:])
        cf.append(corr_data)

    return park_info, vd_info, meta_feature, cf

def float2int(pred):
    temp = []
    for i in pred:
        park_res = int(i[1])
        vd_res = int(i[0])
        if park_res < 0:
            park_res = 0
        if vd_res < 0:
            vd_res = 0

        #temp.append([park_res, vd_res])
        temp.append([vd_res, park_res])
    
    return temp


start_time = datetime.now() #code start time

##----紀錄預測結果用----##
"""
#找停車場站點
park_index_file = 'D:/YZU/Master/paper/old_paper/Xinyi_park_vd/park/park_index.csv'
park_test = pd.read_csv(park_index_file)
park_state_index = park_test.columns[1:]

#找vd站點
vd_index_file = 'D:/YZU/Master/paper/old_paper/Xinyi_park_vd/vd/vd_index.csv'
vd_test = pd.read_csv(vd_index_file)
vd_state_index = vd_test.columns[1:]
"""
##--------------------##

for days in range(31):
    datedata = ['201808' + str(days+1).zfill(2)]

    ##----紀錄預測結果用----##
    """
    #生成預測檔
    station = ['Station']
    park_predict_file = 'D:/YZU/Master/paper/old_paper/Xinyi_park_vd/divide_ver/Prediction/Park' + str(datedata[0]) + '_park.csv'
    if not os.path.isfile(park_predict_file):
        park_file = pd.read_csv('D:/YZU/Master/paper/old_paper/Xinyi_park_vd/park/new_history/201803/new_park_history_20180301.csv', encoding='ISO-8859-1')
        p_id = list(park_file['id'].values)
        p_latlon = list(park_file[['Latitude', 'Longitude']].values)
        p_data = list(park_file['available_car-00:00'].values)
        _, namedata = check.find_map(p_data, p_latlon, p_id, state=True)
        #挑選站點
        new_p_id = []
        for i in park_state_index:
            new_p_id.append(str(namedata[int(i)]))
        
        with open(park_predict_file, 'w', newline='') as pf:
            p_writer = csv.writer(pf)
            p_writer.writerow(station)

            for j in new_p_id:
                p_writer.writerow([j])
    
    vd_predict_file = 'D:/YZU/Master/paper/old_paper/Xinyi_park_vd/divide_ver/Prediction/vd/' + str(datedata[0]) + '_vd.csv'
    if not os.path.isfile(vd_predict_file):
        vd_file = pd.read_csv('D:/YZU/Master/paper/old_paper/Xinyi_park_vd/vd/new_history/信義區VD歷史資料_change_2018-03-01.csv', encoding='ISO-8859-1')
        v_id = list(vd_file['VD_id'].values)
        v_latlon = list(vd_file[['Latitude', 'Longitude']].values)
        v_data = list(vd_file['avgspeed-00:00'].values)
        _, namedata = check.find_map(v_data, v_latlon, v_id, state=False)
        #挑選站點
        new_v_id = []
        for i in vd_state_index:
            new_v_id.append(str(namedata[int(i)]))

        with open(vd_predict_file, 'w', newline='') as vf:
            v_writer = csv.writer(vf)
            v_writer.writerow(station)

            for j in new_v_id:
                v_writer.writerow([j])
    
    pFile = pd.read_csv(park_predict_file, encoding='utf-8')
    park_df = pd.DataFrame(columns=(['Station']))
    park_df = pFile

    vFile = pd.read_csv(vd_predict_file, encoding='utf-8')
    vd_df = pd.DataFrame(columns=(['Station']))
    vd_df = vFile
    """
    ##--------------------##

    all_labels, all_preds = [], []
    park_states_, vd_states_, park_pred_, vd_pred_ = [], [], [], []
    for d in datedata:
        for i in range(24):
            for mins in ['00', '15', '30', '45']:
                i = str(i).zfill(2)
                date = d + str(i) + str(mins)

                pred_park_res, pred_vd_res = [], []
                timestamp = string2timestamp(date)
                print('Now predict {0}'.format(timestamp))

                #取ground truth
                label, park_state, vd_state = get_label(date)
                label = np.reshape(label, (240, 2))
                all_labels.extend(label)
                park_states_.extend(park_state)
                vd_states_.extend(vd_state)

                #根據預測的日期-時間，讀取我們所需的歷史資料
                park_info, vd_info, meta_feature, cf = read_data(date, timestamp)
                p_XC = np.array(park_info[0])
                p_XP = np.array(park_info[1])
                p_XT = np.array(park_info[2])
                v_XC = np.array(vd_info[0])
                v_XP = np.array(vd_info[1])
                v_XT = np.array(vd_info[2])
                meta_feature = np.array(meta_feature)
                cf = np.array([cf])

                g = Graph()
                ckpt_dir = "D:/YZU/Master/paper/old_paper/Xinyi_park_vd/divide_ver/New_Model/" #使用的訓練模型
                with tf.Session() as sess:
                    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
                    g.saver.restore(sess, ckpt.model_checkpoint_path)
                    pred = sess.run([g.x_res], feed_dict={g.p_c_inp_t:p_XC, g.p_p_inp_t:p_XP, g.p_t_inp_t:p_XT, g.v_c_inp_t:v_XC, g.v_p_inp_t:v_XP, g.v_t_inp_t:v_XT, g.external_t:meta_feature, g.correlation_t:cf})
                    
                    pred = reverse_MinMaxNormalization(pred) #將預測結果反推回去，ex: 0.5 -> 990
                    pred = np.reshape(pred, (240, 2))
                    pred = float2int(pred)

                    #get park data
                    park_index_file = 'D:/YZU/Master/paper/old_paper/Xinyi_park_vd/park/park_index.csv'
                    park_test = pd.read_csv(park_index_file)
                    park_state_title = park_test.columns[1:]
                    for s in range(len(park_state_title)):
                        pred_park_res.append(pred[int(park_state_title[s])][1])

                    #get vd data
                    vd_index_file = 'D:/YZU/Master/paper/old_paper/Xinyi_park_vd/vd/vd_index.csv'
                    vd_test = pd.read_csv(vd_index_file)
                    vd_state_title = vd_test.columns[1:]
                    for s in range(len(vd_state_title)):
                        pred_vd_res.append(pred[int(vd_state_title[s])][0])

                    all_preds.extend(pred)
                    park_pred_.extend(pred_park_res)
                    vd_pred_.extend(pred_vd_res)

                    ##----紀錄預測結果用----##
                    res_park, res_vd = [], []
                    """
                    res_park.append(pred_park_res)
                    res_vd.append(pred_vd_res)
                    park_df['available_car-' + str(date[8:-2]) + ':' + str(date[-2:])] = res_park[0]
                    park_df.to_csv(park_predict_file, encoding="big5", na_rep='NA', index=0)

                    vd_df['avgspeed-' + str(date[8:-2]) + ':' + str(date[-2:])] = res_vd[0]
                    vd_df.to_csv(vd_predict_file, encoding="big5", na_rep='NA', index=0)
                    """
                    ##--------------------##
                    

    # print('{} Finished.'.format(datedata[0]))


end_time = datetime.now() #code end time
print('start time: {0}:{1}'.format(start_time.hour, start_time.minute))
print('end time: {0}:{1}'.format(end_time.hour, end_time.minute))
if end_time.hour == start_time.hour:
    print('Program spent {0} mins'.format(end_time.minute - start_time.minute))
else:
    print('Program spent {0} mins'.format((60 + int(end_time.minute)) - int(start_time.minute)))

park_rmse = mape_calcu.rmse_state(park_states_, park_pred_)
vd_rmse = mape_calcu.rmse_state(vd_states_, vd_pred_)
print('RMSE (real vd):{0}, RMSE (real park):{1}'.format(vd_rmse, park_rmse))
park_mape = mape_calcu.mape_state(park_states_, park_pred_)
vd_mape = mape_calcu.mape_state(vd_states_, vd_pred_)
print('MAPE (real vd):{0}, MAPE (real park):{1}'.format(vd_mape, park_mape))