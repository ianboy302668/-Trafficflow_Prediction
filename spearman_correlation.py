#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import os, csv, warnings

warnings.filterwarnings('ignore')

park_path = 'C:/Work/YZU-CS/AIGo/信義區/park_ST-ResNet/new_history/'
vd_path = 'C:/Work/YZU-CS/AIGo/信義區/park_vd_ST-ResNet/new_vd_history/'
corr_csv = 'C:/Work/YZU-CS/AIGo/信義區/park_vd_ST-ResNet/vd_name.csv' #vd站點名稱的csv

#過濾沒有資料的值，ex: -99
def filter_data(x, y):
    xt, yt = [], []
    for i in range(len(x)):
        rec = 0
        for j in range(len(x[i])):
            if str(x[i][j])[0] == '-' or str(y[i][j])[0] == '-':
                rec += 1
        if rec == 0:
            xt.extend(x[i])
            yt.extend(y[i])

    return xt, yt

park_1 = pd.read_csv('C:/Work/YZU-CS/AIGo/信義區/park_ST-ResNet/201803/new_park_history_20180301.csv')
park_1_id = park_1['id'].values #取得地圖上的所有網格的資料
park_name = ['41', '2', '1', '324', '9', '48', '107', '323', '88', '3'] #我們所需要的站點名稱
park_index = []
#找所需站點的index
for pname in park_name:
    park_index.append(list(park_1_id).index(pname))

all_Temp, vd_name = [], []

#讀取一個站點的全部資料(ex:3~8月)，兩個系統的都要取(ex: park and vd)
with open(corr_csv, newline='') as csvfile:
    rows = csv.reader(csvfile)
    for row in rows:
        temp = [row[0]]
        vd_name.append(row[0])

        Temp = []
        for i in range(len(park_index)):
            x, y = [], []
            park_month_file = os.listdir(park_path)
            vd_month_file = os.listdir(vd_path)
            for park_file, vd_file in zip(park_month_file, vd_month_file):
                park_month_path = os.path.join(park_path, park_file)
                vd_month_path = os.path.join(vd_path, vd_file)

                park_day_file = os.listdir(park_month_path)
                vd_day_file = os.listdir(vd_month_path)

                for park_csvfile, vd_csvfile in zip(park_day_file, vd_day_file):
                    park_file_path = os.path.join(park_month_path, park_csvfile)
                    vd_file_path = os.path.join(vd_month_path, vd_csvfile)

                    park_csv = pd.read_csv(park_file_path, encoding='ISO-8859-1')
                    park_time_title = park_csv.columns[4:] #時間點，ex: 00:00, 00:15, 00:20...

                    vd_csv = pd.read_csv(vd_file_path, encoding='ISO-8859-1')
                    vd_id = vd_csv['VD_id'].values
                    vd_time_title = vd_csv.columns[5:101] #時間點，ex: 00:00, 00:15, 00:20...
                    for vname in range(len(vd_id)):
                        if str(vd_id[vname]) == row[0]: #符合的站點名稱
                            vd_index = vname
                    
                    occupy_data = list(park_csv.loc[park_index[i]][4:]) #讀取一個站點當天的全部資料，00:00 ~ 23:45
                    speed_data = list(vd_csv.loc[vd_index][5:101])      #讀取一個站點當天的全部資料，00:00 ~ 23:45
                    x.append(occupy_data)
                    y.append(speed_data)

            xt, yt = filter_data(x, y) #過濾負值的資料，ex: -99
            r, p = spearmanr(xt, yt)   #計算spearman相關係數
            print('VD: {0} - Park: {1}, r: {2}, p: {3}'.format(row[0], park_name[i], r, p))
            Temp.append(r)
        all_Temp.append([Temp])

with open('C:/Work/YZU-CS/AIGo/信義區/park_vd_ST-ResNet/correlation/spearman_corr_Aug.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    title = [' ', '41', '2', '1', '324', '9', '48', '107', '323', '88', '3']
    writer.writerow(title)
    for m in range(len(vd_name)):
        corr_data = [vd_name[m]]
        corr_data.extend(all_Temp[m][0])
        writer.writerow(corr_data)

print('Done.')