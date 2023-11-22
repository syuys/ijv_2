#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 10:51:31 2023

@author: md703
"""

import numpy as np
import pandas as pd
import os
import matplotlib as mpl
import json
import matplotlib.pyplot as plt
plt.close("all")
plt.rcParams.update({"mathtext.default": "regular"})
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["figure.dpi"] = 300


def set_mua_matrix(sessionID, matBas):
    # initialize the mua to be returned
    mua = []
    
    # set mua by existed mua file, matBas = [muapath_1, muapath_2, ...]
    if type(matBas) == list:
        for muaPath in matBas:
            with open(muaPath) as f:
                tmp = json.load(f)
            mua.append([tmp["1: Air"],
                        tmp["2: PLA"],
                        tmp["3: Prism"],
                        tmp["4: Skin"],
                        tmp["5: Fat"],
                        tmp["6: Muscle"],
                        tmp["7: Muscle or IJV (Perturbed Region)"],
                        tmp["8: IJV"],
                        tmp["9: CCA"]
                        ])
        pass
    
    # set mua by different SO2, matBas = (SijvO2_array, SccaO2_array)
    elif type(matBas) == tuple:
        wl = int(sessionID[-5:-2])
        eachSubstanceMua = pd.read_csv("/home/md703/syu/ijv_2/shared_files/model_input_related/optical_properties/coefficients_in_cm-1_OxyDeoxyConc150.csv",
                                       usecols = lambda x: "Unnamed" not in x)
        wlp = eachSubstanceMua["wavelength"].values
        oxy = eachSubstanceMua["oxy"].values * 0.1  # 1/mm
        deoxy = eachSubstanceMua["deoxy"].values * 0.1  # 1/mm
        
        muaHbO2 = np.interp(wl, wlp, oxy)
        muaHb = np.interp(wl, wlp, deoxy)        
        
        SijvO2 = matBas[0]
        SccaO2 = matBas[1]
        muaijvSet = muaHbO2 * SijvO2 + muaHb * (1-SijvO2)
        muaccaSet = muaHbO2 * SccaO2 + muaHb * (1-SccaO2)
        
        # main
        ijvType = sessionID.split("_")[1]
        muaPath = os.path.join(sessionID, f"mua_{wl}nm.json")
        with open(muaPath) as f:
            tmp = json.load(f)
        for muaijv in muaijvSet:
            for muacca in muaccaSet:
                mua.append([tmp["1: Air"],
                            tmp["2: PLA"],
                            tmp["3: Prism"],
                            tmp["4: Skin"],
                            tmp["5: Fat"],
                            tmp["6: Muscle"],
                            muaijv if ijvType == "dis" else tmp["7: Muscle or IJV (Perturbed Region)"],  # pertubed region
                            muaijv,
                            muacca
                            ])        
    
    else:
        raise Exception("matBas setting is weird !")
    
    return np.array(mua).T


def colorFader(c1, c2, mix=0):
    '''
    fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1) 

    '''
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)


def remove_spike(wl, data, normalStdTimes, showTargetSpec):
    mean = data.mean(axis=0)
    std = data.std(ddof=1, axis=0)
    targetSet = []  # save spike idx
    for idx, s in enumerate(data):  # iterate spectrum in every time frame
        isSpike = np.any(abs(s-mean) > normalStdTimes*std)
        if isSpike:
            targetSet.append(idx) 
    print(f"target = {targetSet}")
    if len(targetSet) != 0:
        for target in targetSet:
            # show target spec and replace that spec by using average of the two adjacents
            if showTargetSpec:
                plt.plot(wl, data[target])
                plt.xlabel("Wavelength [nm]")
                plt.ylabel("Intensity [counts]")    
                plt.title(f"spike idx: {target}")
                plt.show()
            
            data[target] = (data[target-1] + data[target+1]) / 2
            
    return data


def get_peak_final(data):
    '''
    Detect peak (rmax and rmin) position in invivo-signal

    '''
    max_idx_set = []
    min_idx_set = []
    data_mean = data.mean()
    
    idx = 0
    break_out_flag = False
    state = data[idx] < data_mean  
    while (data[idx] < data_mean) == state:
        idx += 1
    idx_s = idx
    while True:
        
        state = data[idx] < data_mean
        # while ((data[idx-1] < data_mean) != state) | ((data[idx] < data_mean) == state) | (idx-idx_s <= 3):
        while ((data[idx] < data_mean) == state) | (idx-idx_s <= 3):
        # while (data[idx] < data_mean) == state:
            idx += 1
            if idx >= len(data):
                break_out_flag = True
                break
        if break_out_flag:
            break
        idx_e = idx
        
        minimum_interval = 6
        data_local = data[idx_s:idx_e]
        if data_local.mean() > data_mean:
            max_idx = idx_s + np.argmax(data_local)
            if len(max_idx_set) == 0:
                max_idx_set.append(max_idx)
            elif max_idx - max_idx_set[-1] >= minimum_interval:
                max_idx_set.append(max_idx)
        if data_local.mean() <= data_mean:
            min_idx = idx_s + np.argmin(data_local)
            if len(min_idx_set) == 0:
                min_idx_set.append(min_idx)
            elif min_idx - min_idx_set[-1] >= minimum_interval:
                min_idx_set.append(min_idx)
        
        idx_s = idx_e
    
    max_idx_set, min_idx_set = np.array(max_idx_set), np.array(min_idx_set)    
    
    # # remove improper peak idx
    # data_std = data.std(ddof=1)    
    # max_idx_set = max_idx_set[abs(data[max_idx_set] - data_mean) > (0.5*data_std)]
    # min_idx_set = min_idx_set[abs(data[min_idx_set] - data_mean) > (0.5*data_std)]
    
    # fine-tuning
    for _ in range(5):
        for idx, max_idx in enumerate(max_idx_set):
            start = 0 if max_idx-(minimum_interval-1) < 0 else max_idx-(minimum_interval-1)
            end = max_idx+minimum_interval
            # print((start, end))
            tmp_idx = np.argmax(data[start:end])
            max_idx = start + tmp_idx
            max_idx_set[idx] = max_idx
        for idx, min_idx in enumerate(min_idx_set):
            start = 0 if min_idx-(minimum_interval-1) < 0 else min_idx-(minimum_interval-1)
            end = min_idx+minimum_interval
            # print((start, end))
            tmp_idx = np.argmin(data[start:end])
            min_idx = start + tmp_idx
            min_idx_set[idx] = min_idx
    
    # remove improper peak idx
    # cutbackforthpercent = 0.1
    # max_cutnum = round(len(max_idx_set)*cutbackforthpercent)
    # min_cutnum = round(len(min_idx_set)*cutbackforthpercent)
    # data_max, data_min = data[max_idx_set], data[min_idx_set]
    # sort_max, sort_min = np.argsort(data_max), np.argsort(data_min)
    # max_idx_set = max_idx_set[sort_max][max_cutnum:-max_cutnum]
    # min_idx_set = min_idx_set[sort_min][min_cutnum:-min_cutnum]
    
    max_idx_set.sort()
    min_idx_set.sort()
    
    return np.unique(max_idx_set), np.unique(min_idx_set)


def get_deltaR_by_rmean(rmax, rmin):
    '''
    deltaR means difference between Rmax and Rmin.

    '''
    deltaR = rmax - rmin
    rmean = (rmax+rmin) / 2
    contrast = deltaR / rmean
    return contrast


def get_rmax_by_rmin(rmax, rmin):
    contrast = rmax / rmin
    return contrast


def get_log_Of_rmax_by_rmin(rmax, rmin):
    '''
    log means ln (np.log).

    '''
    contrast = get_rmax_by_rmin(rmax, rmin)
    contrast = np.log(contrast)
    return contrast


def get_deltaRef_by_so2(ref, so2):
    '''
    deltaRef means change percentage of reflectance.

    '''
    deltaRef = np.diff(ref, axis=1) / ref[:, :-1]
    deltaSo2 = np.diff(so2)
    return deltaRef / deltaSo2[None, :]    


def get_deltaContrast_by_so2(rmax, rmin, contrastType, so2):
    deltaSo2 = np.diff(so2)
    # if deltaSo2[0] < 1: deltaSo2 *= 100  # transformed to percentage
    
    if contrastType == "deltaRbyrmean":
        contrast = get_deltaR_by_rmean(rmax, rmin)
        
    elif contrastType == "rmaxbyrmin":
        contrast = get_rmax_by_rmin(rmax, rmin)
        
    elif contrastType == "logOfrmaxbyrmin":
        contrast = get_log_Of_rmax_by_rmin(rmax, rmin)
        
    else:
        raise Exception("Contrast type is not existed !")
    
    # contrast shape: [sds_num, so2_num]
    deltaContrast = np.diff(contrast, axis=1)
    
    metric = deltaContrast / deltaSo2[None, :]
    
    return metric


def normalize(data):
    tmp = data.copy()
    tmp -= 1
    tmp -= tmp.min()
    tmp /= tmp.max()
    return tmp
        