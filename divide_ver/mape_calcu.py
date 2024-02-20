#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
import pandas as pd

#計算整張地圖的RMSE
def rmse(y_true, y_pred):
    park_true_all, vd_true_all, park_pred_all, vd_pred_all = [], [], [], []
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    for i in range(len(y_true)):
        if str(y_true[i][0])[0] != '-':
            park_true_all.append(y_true[i][0])
            park_pred_all.append(y_pred[i][0])
        if str(y_true[i][1])[0] != '-':
            vd_true_all.append(y_true[i][1])
            vd_pred_all.append(y_pred[i][1])

    park_true_all, vd_true_all, park_pred_all, vd_pred_all = np.array(park_true_all), np.array(vd_true_all), np.array(park_pred_all), np.array(vd_pred_all)
    return sqrt(mean_squared_error(park_true_all, park_pred_all)), sqrt(mean_squared_error(vd_true_all, vd_pred_all))

#只計算站點的RMSE
def rmse_state(y_true, y_pred):
    y_true_all, y_pred_all = [], []
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    for i in range(len(y_true)):
        if str(y_true[i])[0] != '-':
            y_true_all.append(y_true[i])
            y_pred_all.append(y_pred[i])

    y_true_all, y_pred_all = np.array(y_true_all), np.array(y_pred_all)
    return sqrt(mean_squared_error(y_true_all, y_pred_all))

#計算整張地圖的MAPE
def mape(y_true, y_pred):
    park_true_all, vd_true_all, park_pred_all, vd_pred_all = [], [], [], []
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    for i in range(len(y_true)):
        if float(y_true[i][0]) != 0.0 and str(y_true[i][0])[0] != '-':
            park_true_all.append(y_true[i][0])
            park_pred_all.append(y_pred[i][0])
        if float(y_true[i][1]) != 0.0 and str(y_true[i][1])[0] != '-':
            vd_true_all.append(y_true[i][1])
            vd_pred_all.append(y_pred[i][1])

    park_true_all, vd_true_all, park_pred_all, vd_pred_all = np.array(park_true_all), np.array(vd_true_all), np.array(park_pred_all), np.array(vd_pred_all)
    return np.mean(np.abs((park_true_all - park_pred_all) / park_true_all)) * 100, np.mean(np.abs((vd_true_all - vd_pred_all) / vd_true_all)) * 100

#只計算站點的MAPE
def mape_state(y_true, y_pred):
    y_true_all, y_pred_all = [], []
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    for i in range(len(y_true)):
        if float(y_true[i]) != 0.0 and str(y_true[i])[0] != '-':
            y_true_all.append(y_true[i])
            y_pred_all.append(y_pred[i])

    y_true_all, y_pred_all = np.array(y_true_all), np.array(y_pred_all)
    return np.mean(np.abs((y_true_all - y_pred_all) / y_true_all)) * 100