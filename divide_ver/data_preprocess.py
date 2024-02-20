#!/usr/bin/python
# -*- coding: utf-8 -*-
import pandas as pd
import os, time, random, math
import numpy as np
from datetime import datetime

np.random.seed(1337) 

get_index = dict()

#字串轉時間格式
def string2timestamp(strings, T=96):
    timestamps = []
    for t in strings: #strings裡面為20180301-00:00形式
        year, month, day, hour, tm_min = int(t[:4]), int(t[4:6]), int(t[6:8]), int(t[9:11]), int(t[12:])
        timestamps.append(pd.Timestamp(datetime(year, month, day, hour, tm_min))) #轉成2018-03-01 00:00形式
        
    return timestamps

#把資料縮放到0~1之間
def MinMaxNormalization(X):
    X_train = X[:-240]
    #x_min = X_train.min()
    x_min = 0
    x_max = X_train.max()
    print('min:{0}, max:{1}'.format(x_min, x_max))

    X = 1. * (X - x_min) / (x_max - x_min)

    return X

def make_index(timestamps_y):
    global get_index
    
    timestamps_y = string2timestamp(timestamps_y)
    for i, ts in enumerate(timestamps_y):
        get_index[ts] = i

def get_matrix(x_all, timestamps):
    global get_index
    return x_all[get_index[timestamps]]

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

def revise_data_shape(x):
    x1, x2 = x[0], x[1]
    column = []
    for i in range(len(x1)):
        row = []
        for j in range(len(x1[i])):
            row.append([x1[i][j], x2[i][j]])
        column.append(row)
    x_final = column.copy()
    
    return x_final

#過濾有遺失值的資料
def create_dataset(x_all, timestamps_y, T=96, len_closeness=3, len_period=1, len_trend=1, PeriodInterval=1, TrendInterval=7):
    offset_frame = pd.DateOffset(minutes=24 * 60 // T)
    
    timestamps_y = string2timestamp(timestamps_y)
    
    XC, XP, XT, Y, timestamps_Y = [], [], [], [], []
    #找出closeness, period, trend的時間長度
    depends = [list(range(1, len_closeness+1)), [PeriodInterval * T * j for j in range(1, len_period+1)], [TrendInterval * T * j for j in range(1, len_trend+1)]]
    i = max(T * TrendInterval * len_trend, T * PeriodInterval * len_period, len_closeness)
    
    #不考慮c, p, t中有一個遺失值的資料，要過濾
    while i < len(timestamps_y):
        x_c = [get_matrix(x_all, timestamps_y[i] - j * offset_frame) for j in depends[0]]
        x_c_final = revise_shape(x_c)
        cc = 0
        for col in range(len(x_c_final)):
            for row in range(len(x_c_final[col])):
                for num in range(len(x_c_final[col][row])):
                    if str(x_c_final[col][row][num])[0] == '-':
                        cc += 1
        #print(cc)

        x_p = [get_matrix(x_all, timestamps_y[i] - j * offset_frame) for j in depends[1]]
        x_p_final = revise_shape(x_p)
        pp = 0
        for col in range(len(x_p_final)):
            for row in range(len(x_p_final[col])):
                for num in range(len(x_p_final[col][row])):
                    if str(x_p_final[col][row][num])[0] == '-':
                        pp += 1
        #print(pp)

        x_t = [get_matrix(x_all, timestamps_y[i] - j * offset_frame) for j in depends[2]]
        x_t_final = revise_shape(x_t)
        tt = 0
        for col in range(len(x_t_final)):
            for row in range(len(x_t_final[col])):
                for num in range(len(x_t_final[col][row])):
                    if str(x_t_final[col][row][num])[0] == '-':
                        tt += 1
        #print(tt)

        y = get_matrix(x_all, timestamps_y[i])
        #y_final = revise_shape(y)

        if cc == 0 and pp == 0 and tt == 0:
            if len_closeness > 0:
                XC.append(x_c_final)
            if len_period > 0:
                XP.append(x_p_final)
            if len_trend > 0:
                XT.append(x_t_final)
            #Y.append(y_final)
            Y.append(y)
            timestamps_Y.append(timestamps_y[i])
        i += 1
    
    return XC, XP, XT, Y, timestamps_Y

#external factor轉one-hot encoding
def timestamp2vec(timestamps):
    vec = [time.strptime(str(t.date().strftime('%Y') + t.date().strftime('%m') + t.date().strftime('%d')), '%Y%m%d').tm_wday for t in timestamps]
    #vec = [time.strptime(str(t[:8], encoding='utf-8'), '%Y%m%d').tm_wday for t in timestamps]
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

"""
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
"""

#把資料resize到地圖的大小，因為有時候一個grid有兩個站點，在csv中會分別顯示資料
#調整grid
def checkvalue(x, latlon, name, state):
    mapdata, latlondata, namedata = [], [], []
    for i in range(len(x)):
        Index = True
        if i >= 2:
            #跳過同網格內的點
            if latlon[i][0] == latlon[i-1][0] and latlon[i][1] == latlon[i-1][1]:
                Index = False
        if Index:
            #print('index:{0}'.format(latlon[i]))
            #檢查同一個網格內是否有多個停車場
            if i != (len(x) - 1):
                #經緯度一樣
                if latlon[i][0] == latlon[i+1][0] and latlon[i][1] == latlon[i+1][1]:
                    if state == True: #park
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
                            latlondata.append(latlon[i])
                            namedata.append(name[i])
                        elif num == 1:
                            if s1 == 1:
                                mapdata.append(x[i+1])
                                latlondata.append(latlon[i+1])
                                namedata.append(name[i+1])
                            elif s2 == 1:
                                mapdata.append(x[i])
                                latlondata.append(latlon[i])
                                namedata.append(name[i])
                        elif num == 0:
                            temp = x[i] + x[i+1]
                            mapdata.append(temp)
                            latlondata.append(latlon[i])
                            namedata.append(name[i])
                    else: #vd
                        temp = math.ceil((x[i] + x[i+1])/2)
                        x[i] = temp
                        mapdata.append(x[i])
                        latlondata.append(latlon[i])
                        namedata.append(name[i])
                #經緯度不一樣
                else:
                    if str(x[i]) == 'nan':
                        x[i] = 0
                        mapdata.append(x[i])
                        latlondata.append(latlon[i])
                        namedata.append(name[i])
                    else:
                        mapdata.append(x[i])
                        latlondata.append(latlon[i])
                        namedata.append(name[i])
            elif i == (len(x) - 1):
                if str(x[i]) == 'nan':
                    x[i] = 0
                    mapdata.append(x[i])
                    latlondata.append(latlon[i])
                    namedata.append(name[i])
                else:
                    mapdata.append(x[i])
                    latlondata.append(latlon[i])
                    namedata.append(name[i])

    return mapdata

def deal_with_data(x_all, timestamps_y, len_closeness=3, len_period=1, len_trend=1, meta_data=True):
    xc, xp, xt, y = [], [], [], []

    x_all = [x_all]
    timestamps_y = [timestamps_y]
    x_all_final = []
    timestamps_Y = []
    #把資料做min-max normalization
    for d in x_all:
        d = np.asarray(d)
        x_all_final.append(MinMaxNormalization(d))

    #將資料做成(c, p, t)格式
    for data, timestamps in zip(x_all_final, timestamps_y):
        _XC, _XP, _XT, _Y, _timestamps_Y = create_dataset(data, timestamps, len_closeness=3, len_period=1, len_trend=1)
        for i in [_XC, _XP, _XT, _Y]:
            i = np.array(i)
            print(i.shape)
        print('=' * 20)
        xc.append(_XC)
        xp.append(_XP)
        xt.append(_XT)
        y.append(_Y)
        timestamps_Y += _timestamps_Y
    
    xc = np.vstack(xc)
    xp = np.vstack(xp)
    xt = np.vstack(xt)
    y = np.vstack(y)

    return xc, xp, xt, y, timestamps_Y

#把資料分成train跟validation，並打亂順序
def divide_data(xc, xp, xt, y, timestamps_Y, corr=None, len_closeness=3, len_period=1, len_trend=1, meta_data=True):
    xc = np.asarray(xc)
    xp = np.asarray(xp)
    xt = np.asarray(xt)
    y = np.asarray(y)
    timestamps_Y = np.asarray(timestamps_Y)

    len_test = 24 * 10
    #分train, test
    xc_train, xp_train, xt_train, y_train = xc[:-len_test], xp[:-len_test], xt[:-len_test], y[:-len_test]
    xc_test, xp_test, xt_test, y_test = xc[-len_test:], xp[-len_test:], xt[-len_test:], y[-len_test:]
    timestamp_train, timestamp_test = timestamps_Y[:-len_test], timestamps_Y[-len_test:]
    #timestamp_train_str = timestamp2string(timestamp_train)
    #timestamp_test_str = timestamp2string(timestamp_test)

    train_num_example = np.array(xc_train).shape[0]
    test_num_example = np.array(xc_test).shape[0]
    train_arr = np.arange(train_num_example)
    test_arr = np.arange(test_num_example)
    np.random.shuffle(train_arr)
    np.random.shuffle(test_arr)
    xc_train, xp_train, xt_train, y_train, timestamp_train = xc_train[train_arr], xp_train[train_arr], xt_train[train_arr], y_train[train_arr], timestamp_train[train_arr]
    xc_test, xp_test, xt_test, y_test, timestamp_test = xc_test[test_arr], xp_test[test_arr], xt_test[test_arr], y_test[test_arr], timestamp_test[test_arr]

    """
    #write to text file for check
    fp1 = open('traindata.txt', 'w')
    fp1.writelines(timestamp_train_str)
    fp1.close()

    fp2 = open('testdata.txt', 'w')
    fp2.writelines(timestamp_test_str)
    fp2.close()
    """
    
    x_train, x_test = [], []
    for l, x_ in zip([len_closeness, len_period, len_trend], [xc_train, xp_train, xt_train]):
        if l > 0:
            x_train.append(x_)
    for l, x_ in zip([len_closeness, len_period, len_trend], [xc_test, xp_test, xt_test]):
        if l > 0:
            x_test.append(x_)

    #load meta feature
    if meta_data:
        meta_feature = timestamp2vec(timestamps_Y)
        metadata_dim = meta_feature.shape[1]
        meta_feature_train, meta_feature_test = meta_feature[:-len_test], meta_feature[-len_test:]

    #load correlation feature
    if corr == True:
        corr_path = 'D:/YZU/Master/paper/old_paper/Xinyi_park_vd/spearman_corr_.csv' # vd->park

        corr_csv = pd.read_csv(corr_path, encoding='ISO-8859-1')
        Corr, corr_feature = [], []
        
        for i in range(27):
            corr_data = list(corr_csv.loc[i][1:])
            Corr.append(corr_data)

        for i in range(meta_feature.shape[0]):
            corr_feature.append(Corr)

        cf_train, cf_test = corr_feature[:-len_test], corr_feature[-len_test:]

        return x_train, y_train, x_test, y_test, timestamp_train, timestamp_test, metadata_dim, meta_feature_train, meta_feature_test, cf_train, cf_test
    else:
        return x_train, y_train, x_test, y_test, timestamp_train, timestamp_test, metadata_dim, meta_feature_train, meta_feature_test

def load_data(len_closeness=3, len_period=1, len_trend=1, meta_data=True):
    global get_index

    park_path = 'D:/YZU/Master/paper/old_paper/Xinyi_park_vd/park/new_history/'
    vd_path = 'D:/YZU/Master/paper/old_paper/Xinyi_park_vd/vd/new_history/'
    
    park_all, vd_all = [], []
    timestamps_y = []
    
    park_month_file = os.listdir(park_path) #讀取資料夾的停車場資料
    vd_month_file = os.listdir(vd_path) #讀取資料夾的VD資料

    #依序讀取每個月的資料
    for park_file, vd_file in zip(park_month_file, vd_month_file):
        park_month_path = os.path.join(park_path, park_file)
        vd_month_path = os.path.join(vd_path, vd_file)
        
        park_day_file = os.listdir(park_month_path)
        vd_day_file = os.listdir(vd_month_path)

        for park_csvfile, vd_csvfile in zip(park_day_file, vd_day_file):
            park_file_path = os.path.join(park_month_path, park_csvfile)
            vd_file_path = os.path.join(vd_month_path, vd_csvfile)

            Date = str(park_csvfile.split('_')[3].split('.')[0])

            #park data
            park_csv = pd.read_csv(park_file_path, encoding='ISO-8859-1')
            park_latlon = park_csv[['Latitude', 'Longitude']].values
            park_id = park_csv['id'].values
            park_time_title = park_csv.columns[4:]

            #vd data
            vd_csv = pd.read_csv(vd_file_path, encoding='ISO-8859-1')
            vd_latlon = vd_csv[['Latitude', 'Longitude']].values
            vd_id = vd_csv['VD_id'].values
            vd_time_title = vd_csv.columns[5:101]
            
            for t in range(0, len(park_time_title)):
                Date_time = Date + '-' + str(park_time_title[t].split('-')[1])
                
                #park data
                occupy_data = list(park_csv[park_time_title[t]].values)
                occupy_data = checkvalue(occupy_data, park_latlon, park_id, state=True)
                print(len(occupy_data)) # debug
                occupy_data = np.reshape(occupy_data, (15, 16))

                #vd data
                avgspeed_data = list(vd_csv[vd_time_title[t]].values)
                avgspeed_data = checkvalue(avgspeed_data, vd_latlon, vd_id, state=False)
                avgspeed_data = np.reshape(avgspeed_data, (15, 16))

                #data_temp = [occupy_data, avgspeed_data]
                #data_temp = revise_data_shape(data_temp)

                park_all.append(occupy_data)
                vd_all.append(avgspeed_data)
                timestamps_y.append(Date_time)
    
    park_all = np.array(park_all)
    print('park data size: ' + str(park_all.shape))
    vd_all = np.array(vd_all)
    print('vd data size: ' + str(vd_all.shape))

    make_index(timestamps_y)

    #處理停車場
    park_xc, park_xp, park_xt, park_y, park_timestamps_Y = deal_with_data(park_all, timestamps_y, len_closeness=3, len_period=1, len_trend=1, meta_data=True)
    #處理VD
    vd_xc, vd_xp, vd_xt, vd_y, vd_timestamps_Y = deal_with_data(vd_all, timestamps_y, len_closeness=3, len_period=1, len_trend=1, meta_data=True)

    final_park_xc, final_park_xp, final_park_xt, final_park_y = [], [], [], []
    final_vd_xc, final_vd_xp, final_vd_xt, final_vd_y = [], [], [], []

    #篩選兩邊的資料，選取一天當中同時有完整的停車場與VD資料
    final_timestamps_Y = list(set(park_timestamps_Y) & set(vd_timestamps_Y))
    for i in final_timestamps_Y:
        park_data_index = list(park_timestamps_Y).index(i)
        vd_data_index = list(vd_timestamps_Y).index(i)

        final_park_xc.append(park_xc[park_data_index])
        final_park_xp.append(park_xp[park_data_index])
        final_park_xt.append(park_xt[park_data_index])
        final_park_y.append(park_y[park_data_index])

        final_vd_xc.append(vd_xc[vd_data_index])
        final_vd_xp.append(vd_xp[vd_data_index])
        final_vd_xt.append(vd_xt[vd_data_index])
        final_vd_y.append(vd_y[vd_data_index])

    p_x_train, p_y_train, p_x_test, p_y_test, p_timestamp_train, p_timestamp_test, p_meta_dim, p_meta_train, p_meta_test, cf_train, cf_test = divide_data(final_park_xc, final_park_xp, final_park_xt, final_park_y, final_timestamps_Y, corr=True)
    v_x_train, v_y_train, v_x_test, v_y_test, v_timestamp_train, v_timestamp_test, v_meta_dim, v_meta_train, v_meta_test = divide_data(final_vd_xc, final_vd_xp, final_vd_xt, final_vd_y, final_timestamps_Y)

    park_info = [p_x_train, p_y_train, p_x_test, p_y_test, p_timestamp_train, p_timestamp_test, p_meta_dim, p_meta_train, p_meta_test]
    vd_info = [v_x_train, v_y_train, v_x_test, v_y_test, v_timestamp_train, v_timestamp_test, v_meta_dim, v_meta_train, v_meta_test]
    corr_info = [cf_train, cf_test]

    return park_info, vd_info, corr_info


"""
park_info, vd_info, corr_1_info, corr_2_info = load_data(len_closeness=3, len_period=, len_trend=1)
for x in park_info[0]:
    x = np.array(x)
    print('park: ' + str(x.shape))
for x in vd_info[0]:
    x = np.array(x)
    print('vd: ' + str(x.shape))
"""