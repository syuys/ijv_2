#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  2 16:40:27 2023

@author: md703
"""

import numpy as np
import pandas as pd
from PyEMD import EMD 
from scipy.signal import find_peaks_cwt, find_peaks, convolve
import json
import matplotlib as mpl
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
plt.rcParams.update({"mathtext.default": "regular"})
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["figure.dpi"] = 300


# %% function

# for signal-denoise, extract motion artifact
def denoise(ori_spec):
    imfs = EMD().emd(ori_spec)
    # artifact = imfs[-1] - imfs[-1].mean()
    imfs[-1] -= imfs[-1].mean()
    artifact = imfs[-3:]  # get the last 3 long-period artifact
    spec_denoise = ori_spec.copy()
    for art in artifact:
        spec_denoise -= art
    
    # plot decomposition detail
    # fig, ax = plt.subplots(imfs.shape[0]+1, 1, figsize=(13, 8))
    # ax[0].plot(ori_spec, 'r')
    # ax[0].set_title(name + ", raw")
    # for n, imf in enumerate(imfs):
    #     ax[n+1].plot(imf, 'g')
    #     ax[n+1].set_title("imf " + str(n+1))        
    # plt.xlabel("time [frame]")
    # plt.tight_layout()
    # plt.show()
    
    return spec_denoise, artifact

# find peak in signal - can compare with scipy.signal.find_peaks function
def get_peak(data, p_start, win):
    idx = []
    p = p_start
    while p+win < len(data):
        argmax = np.argmax(data[p:p+win])
        idx.append(p+argmax)
        p += win
    return np.array(idx)

def colorFader(c1, c2, mix=0): # fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)
    

# %% load data

# wavelength
spec = pd.read_csv("det_1.csv")
wl = np.array(spec.columns[1:].values, dtype=float) # wavelength

# background
bg = pd.read_csv("background.csv")
bg = bg.iloc[370:-10, 1:]
bg = np.array(bg)
# for s in bg:
#     plt.plot(wl, s)
# plt.xlabel("wl [nm]")
# plt.ylabel("reflectance [-]")    
# plt.title("background")
# plt.show()

# experiment
signal = {}
nameSet = ["det_1.csv", "det_2.csv", "det_3.csv", "det_4.csv", "det_5.csv"]
frameStart = 10
frameEnd = 310
for name in nameSet:
    title = name.split(".")[0]
    spec = pd.read_csv(name)
    print(f"{title}: {spec.shape} → ", end="")
    spec = spec.iloc[frameStart:frameEnd, 1:]   # drop first and last 10 time frames, drop first column (time stamp)
    spec = np.array(spec)
    print(spec.shape, end=", ")
    
    # plot raw signal
    # for s in spec:
    #     plt.plot(wl, s)
    # plt.xlabel("wl [nm]")
    # plt.ylabel("reflectance [-]")    
    # plt.title(title)
    # plt.show()
    
    # subtract background and detect spike
    spec -= bg.mean(axis=0)    
    mean = spec.mean(axis=0)
    std = spec.std(ddof=1, axis=0)
    targetSet = []  # save spike idx
    for idx, s in enumerate(spec):  # iterate spectrum in every time frame
        isSpike = np.any(abs(s-mean) > 5*std)
        if isSpike:
            targetSet.append(idx) 
    print(f"target = {targetSet}")
    if len(targetSet) != 0:
        for target in targetSet:
            # replace the spectrum with spike by using average of two adjacent specs
            spec[target] = (spec[target-1] + spec[target+1]) / 2
    
    # save signal with removal of background and spike
    signal[title] = spec
    
    # plot processed signal
    # for s in spec:
    #     plt.plot(wl, s)
    # plt.xlabel("wl [nm]")
    # plt.ylabel("reflectance [-]")
    
    # plt.title(title)
    # plt.show()


# %% extract motion artifact and Rmax, Rmin
maxref_raw_all = []
minref_raw_all = []
contrast_raw_all = []
maxref_denoise_all = []
minref_denoise_all = []
contrast_denoise_all = []
contrast_wl_all = []
live_denoise_all = []
live_max_denoise_all = []
live_min_denoise_all = []
max_index_num_all = []
min_index_num_all = []
width = 8  # for find_peaks_cwt
window = [10, 10, 10, 10, 10] # for get_peak
maxShift = [0, 4, 5, 0, 0]
minShift = [4, 5, 4, 4, 4]
slide = 23
obsInv = 50  # for observing Rmax/Rmin variation within one experiment
integrationTime = 0.1
c1 = "red"
c2 = "blue"
targetWl = [730, 760, 780, 810, 850]

time = np.arange(0, (frameEnd-frameStart)*integrationTime, integrationTime)
pivot = (slide-1) // 2
wlpivot = wl[pivot: -pivot]
for idx, (name, spec) in enumerate(signal.items()):
    
    # detect motion artifact and remove
    spec_time_raw = spec.mean(axis=1)  # do average w.r.t wavelength
    spec_time_denoise, artifact = denoise(spec_time_raw)
    
    
    # CVP
    cvp = spec.sum(axis=1)
    cvp = 1 / cvp
    # plt.figure(figsize=(12, 4))    
    # plt.plot(time, cvp)
    # plt.grid()
    # plt.xlabel(f"time [s], integration time = {integrationTime} s")
    # plt.ylabel("1 / total R  [-]")
    # plt.title(f"{name} - CVP waveform")
    # plt.show()
    
    
    # pattern of radom-select wavelength
    spec_mov = spec.copy()
    spec_mov = convolve(spec_mov, np.ones((1, slide))/slide, mode='valid')
    # plt.figure(figsize=(12, 4))
    # rndSet = [100, 300, 500, 700, 900]
    # for rnd in rndSet:
    #     plt.plot(time, spec_mov[:, rnd], label=f"{wlpivot[rnd]} nm")
    # plt.grid()
    # plt.legend()
    # plt.xlabel(f"time [s], integration time = {integrationTime} s")
    # plt.ylabel("reflectance [-]")
    # plt.title(f"{name} - pattern of different wavelength")
    # plt.show()
    
    
    # raw
    # max_index = find_peaks_cwt(spec_time_raw, np.arange(1, width+0.5))
    # min_index = find_peaks_cwt(-spec_time_raw, np.arange(1, width+0.5))
    max_index = get_peak(spec_time_raw, maxShift[idx], window[idx])
    min_index = get_peak(-spec_time_raw, 0+minShift[idx], window[idx])
    
    # plt.figure(figsize=(12, 4))
    # plt.plot(time, spec_time_raw)
    # plt.scatter((max_index)*integrationTime, spec_time_raw[max_index], label='max')
    # plt.scatter((min_index)*integrationTime, spec_time_raw[min_index], label='min')
    # plt.title(f"{name} - raw")
    # plt.legend()
    # plt.grid()
    # plt.xlabel(f"time [s], integration time = {integrationTime} s")
    # plt.ylabel("reflectance [-]")
    # xlim = plt.xlim()
    # ylim = plt.ylim()
    # plt.show()    
    
    t = 0
    maxref_raw = []
    minref_raw = []
    maxbymin_raw = []
    while t+obsInv <= len(spec_time_raw):
        max_local = max_index[(max_index>=t) & (max_index<t+obsInv)]
        min_local = min_index[(min_index>=t) & (min_index<t+obsInv)]
        spec_time_raw_local_max = spec_time_raw[max_local].mean()
        spec_time_raw_local_min = spec_time_raw[min_local].mean()
        maxref_raw.append(spec_time_raw_local_max)
        minref_raw.append(spec_time_raw_local_min)
        maxbymin_raw.append(spec_time_raw_local_max / spec_time_raw_local_min)
        t += obsInv
    maxref_raw_all.append(maxref_raw)
    minref_raw_all.append(minref_raw)
    contrast_raw_all.append(maxbymin_raw)
    
    # plt.figure(figsize=(12, 4))
    # plt.plot(np.arange((obsInv)*integrationTime, 
    #                     (t+obsInv)*integrationTime, 
    #                     (obsInv)*integrationTime), 
    #           maxbymin_raw, 
    #           "-o")
    # plt.xlim(xlim)
    # ylim_c = plt.ylim()
    # plt.grid()
    # plt.xlabel("time [s]")
    # plt.ylabel("Rmax / Rmin  [-]")
    # plt.title(f"{name} - raw, Rmax/Rmin variation")
    # plt.show()
    
    
    # denoise
    # max_index = find_peaks_cwt(spec_time_denoise, np.arange(1, width))
    # min_index = find_peaks_cwt(-spec_time_denoise, np.arange(1, width))
    max_index = get_peak(spec_time_denoise, maxShift[idx], window[idx])
    min_index = get_peak(-spec_time_denoise, 0+minShift[idx], window[idx])
    max_index_num_all.append(len(max_index))
    min_index_num_all.append(len(min_index))
    
    plt.figure(figsize=(12, 4))
    plt.plot(time, spec_time_denoise)
    plt.scatter((max_index)*integrationTime, spec_time_denoise[max_index], label='max')
    plt.scatter((min_index)*integrationTime, spec_time_denoise[min_index], label='min')
    plt.title(f"{name} - denosied")
    plt.legend()
    plt.grid()
    # plt.ylim(ylim)
    plt.xlabel(f"time [s], integration time = {integrationTime} s")
    plt.ylabel("reflectance [-]")
    plt.show()
    
    t = 0
    maxref_denoise = []
    minref_denoise = []
    maxbymin_denoise = []
    while t+obsInv <= len(spec_time_denoise):
        max_local = max_index[(max_index>=t) & (max_index<t+obsInv)]
        min_local = min_index[(min_index>=t) & (min_index<t+obsInv)]
        spec_time_denoise_local_max = spec_time_denoise[max_local].mean()
        spec_time_denoise_local_min = spec_time_denoise[min_local].mean()
        maxref_denoise.append(spec_time_denoise_local_max)
        minref_denoise.append(spec_time_denoise_local_min)
        maxbymin_denoise.append(spec_time_denoise_local_max / spec_time_denoise_local_min)
        t += obsInv
    maxref_denoise_all.append(maxref_denoise)
    minref_denoise_all.append(minref_denoise)
    contrast_denoise_all.append(maxbymin_denoise)
    
    # plt.figure(figsize=(12, 4))
    # plt.plot(np.arange((obsInv)*integrationTime, 
    #                     (t+obsInv)*integrationTime, 
    #                     (obsInv)*integrationTime), 
    #           maxbymin_denoise, 
    #           "-o")
    # plt.xlim(xlim)
    # plt.ylim(ylim_c)
    # plt.grid()
    # plt.xlabel("time [s]")
    # plt.ylabel("Rmax / Rmin  [-]")
    # plt.title(f"{name} - denoised, Rmax/Rmin variation")
    # plt.show()
    
    
    # live, live_max, live_min
    live = spec.copy()
    for art in artifact:
        live -= art.reshape(-1, 1)
    live = convolve(live, np.ones((1, slide))/slide, mode='valid')
    live_denoise_all.append(live)  # store denoised spectrum
    live_max = live[max_index].mean(axis=0)
    live_min = live[min_index].mean(axis=0)
    
    # plot average
    # fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    # ax.plot(wlpivot, live_max, label="max")
    # ax.plot(wlpivot, live_min, label="min")
    # ax.legend()
    # ax.set_xlabel("wl [nm]")
    # ax.set_ylabel("reflectance [-]")    
    # ax.set_title(f"{name} result - average")
    # plt.show()
    
    # plot each time interval
    t = 0
    contrast_wl = []
    # plt.figure(figsize=(12, 4))
    while t+obsInv <= len(spec_time_denoise):
        max_local = max_index[(max_index>=t) & (max_index<t+obsInv)]
        min_local = min_index[(min_index>=t) & (min_index<t+obsInv)]
        live_max_local = live[max_local].mean(axis=0)
        live_min_local = live[min_local].mean(axis=0)
        live_max_denoise_all.append(live_max_local)
        live_min_denoise_all.append(live_min_local)
        contrast_wl.append(np.interp(targetWl, wlpivot, live_max_local) / np.interp(targetWl, wlpivot, live_min_local))
    #     plt.plot(wlpivot, live_max_local, 
    #              color=colorFader(c1, c2, (t//obsInv)/5), 
    #              linestyle="-", label=f"max, {int(t*integrationTime)}s - {int((t+obsInv)*integrationTime)}s")
    #     plt.plot(wlpivot, live_min_local, 
    #              color=colorFader(c1, c2, (t//obsInv)/5), 
    #              linestyle=":", label="min")
        
        t += obsInv
    
    contrast_wl_all.append(np.array(contrast_wl))
        
    # plt.xlabel("wl [nm]")
    # plt.ylabel("reflectance [-]")
    # plt.legend()
    # plt.title(f"{name} - live")
    # plt.show()

maxref_raw_all = np.array(maxref_raw_all)
minref_raw_all = np.array(minref_raw_all)
contrast_raw_all = np.array(contrast_raw_all)
maxref_denoise_all = np.array(maxref_denoise_all)
minref_denoise_all = np.array(minref_denoise_all)
contrast_denoise_all = np.array(contrast_denoise_all)
 
   
# plot variation of raw signal - reflectance
colorSet = ["#1f77b4", "#ff7f0e"]
for idx, p in enumerate(maxref_raw_all):
    plt.scatter(np.repeat(idx, len(p)), p, marker=".", color=colorSet[0])
means = maxref_raw_all.mean(axis=1)
stds = maxref_raw_all.std(axis=1, ddof=1)
mins = abs(maxref_raw_all.min(axis=1) - means)
maxs = abs(maxref_raw_all.max(axis=1) - means)
# plt.errorbar(np.arange(len(means)), means, yerr=[mins, maxs], 
#                 fmt="-o", capsize=4, label="max with range")
plt.errorbar(np.arange(len(means)), means, yerr=stds, 
              fmt="-", color=colorSet[0], capsize=4, label='max ± σ')

for idx, p in enumerate(minref_raw_all):
    plt.scatter(np.repeat(idx, len(p)), p, marker=".", color=colorSet[1])
means = minref_raw_all.mean(axis=1)
stds = minref_raw_all.std(axis=1, ddof=1)
mins = abs(minref_raw_all.min(axis=1) - means)
maxs = abs(minref_raw_all.max(axis=1) - means)
# plt.errorbar(np.arange(len(means)), means, yerr=[mins, maxs], 
#               fmt="-o", capsize=4, label="min with range")
plt.errorbar(np.arange(len(means)), means, yerr=stds, 
              fmt="-", color=colorSet[1], capsize=4, label='min ± σ')
plt.xticks(np.arange(len(means)))
plt.xlabel("repeat index [-]")
plt.ylabel("reflectance [-]")
plt.legend()
plt.title("stability test (5 repeats), reflectance - raw")
plt.show()


# plot variation of denoise signal - reflectance
for idx, p in enumerate(maxref_denoise_all):
    plt.scatter(np.repeat(idx, len(p)), p, marker=".", color=colorSet[0])
means = maxref_denoise_all.mean(axis=1)
stds = maxref_denoise_all.std(axis=1, ddof=1)
mins = abs(maxref_denoise_all.min(axis=1) - means)
maxs = abs(maxref_denoise_all.max(axis=1) - means)
# plt.errorbar(np.arange(len(means)), means, yerr=[mins, maxs], 
#               fmt="-o", capsize=4, label="max with range")
plt.errorbar(np.arange(len(means)), means, yerr=stds, 
              fmt="-", color=colorSet[0], capsize=4, label='max ± σ')

for idx, p in enumerate(minref_denoise_all):
    plt.scatter(np.repeat(idx, len(p)), p, marker=".", color=colorSet[1])
means = minref_denoise_all.mean(axis=1)
stds = minref_denoise_all.std(axis=1, ddof=1)
mins = abs(minref_denoise_all.min(axis=1) - means)
maxs = abs(minref_denoise_all.max(axis=1) - means)
# plt.errorbar(np.arange(len(means)), means, yerr=[mins, maxs], 
#               fmt="-o", capsize=4, label="min with range")
plt.errorbar(np.arange(len(means)), means, yerr=stds, 
              fmt="-", color=colorSet[1], capsize=4, label='min ± σ')
plt.xticks(np.arange(len(means)))
plt.xlabel("repeat index [-]")
plt.ylabel("reflectance [-]")
plt.legend()
plt.title("stability test (5 repeats), reflectance - denoised")
plt.show()


# analyze live spectrum
# live_denoise_all = np.array(live_denoise_all)
# live_denoise_all = live_denoise_all.reshape(live_denoise_all.shape[0], 
#                                             -1, obsInv, 
#                                             live_denoise_all.shape[-1])
# sample = live_denoise_all.mean(axis=2)
# within_group_mean = sample.mean(axis=1)
# within_group_std = sample.std(ddof=1, axis=1)
# for idx in range(within_group_mean.shape[0]):
#     plt.figure(figsize=(12, 4))
#     plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
#     plt.plot(wlpivot, within_group_std[idx]/within_group_mean[idx])
#     plt.xlabel("wl [nm]")
#     plt.ylabel("cv [-]")
#     plt.title(f"det_{idx} - variation between {sample.shape[1]} intervals")
#     plt.show()
# between_group_mean = sample.mean(axis=0)
# between_group_std = sample.std(ddof=1, axis=0)
# for idx in range(between_group_mean.shape[0]):
#     plt.figure(figsize=(12, 4))
#     plt.plot(wlpivot, between_group_mean[idx], label="mean ± σ")
#     plt.fill_between(wlpivot, 
#                       between_group_mean[idx]-between_group_std[idx], 
#                       between_group_mean[idx]+between_group_std[idx], 
#                       alpha=0.4)
#     plt.legend()
#     plt.xlabel("wl [nm]")
#     plt.ylabel("reflectance [-]")
#     plt.title(f"interval_{idx} - variation between {sample.shape[0]} repeats")
#     plt.show()
# fig, ax = plt.subplots(1, 1, figsize=(12, 4))
# lns = []
# axtmp = ax.twinx()
# lns += ax.plot(wlpivot, between_group_std.mean(axis=0), label="(left) average of between-repeats-std over 6 intervals")
# lns += axtmp.plot(wlpivot, within_group_std.mean(axis=0), label="(right) average of between-intervals-std over 5 repeats")
# lns += ax.plot(wlpivot, between_group_std.mean(axis=0)-within_group_std.mean(axis=0), 
#                 label="(left) difference - noise from detach and reattach detector")
# labels = [l.get_label() for l in lns]
# ax.legend(lns, labels)
# ax.set_xlabel("wl [nm]")
# ax.set_ylabel("std value [-]")
# axtmp.set_ylabel("std value [-]")
# ax.set_title("Result")
# plt.show()


# analyze max, min spectrum
def get_expected_ratio_var(rmax_raw, rmin_raw, corr=False):
    if rmax_raw.ndim == 2:
        rmax_raw, rmin_raw = rmax_raw[None, :, :], rmin_raw[None, :, :]
    r = np.zeros((rmax_raw.shape[0], rmax_raw.shape[-1]), dtype=float)
    if corr:
        for order, (rmax, rmin) in enumerate(zip(rmax_raw, rmin_raw)):
            a = rmax.T
            b = rmin.T
            num = 100
            pivot = 0
            length = a.shape[0]
            while True:
                if pivot + num <= length:
                    aa, bb = a[pivot:pivot+num], b[pivot:pivot+num]
                    r_mat = np.corrcoef(aa, bb)
                    for idx in range(num):
                        r[order, pivot+idx] = r_mat[idx, idx+num]                    
                    pivot += num
                else:
                    num = length - pivot
                    aa, bb = a[pivot:pivot+num], b[pivot:pivot+num]
                    r_mat = np.corrcoef(aa, bb)
                    for idx in range(num):
                        r[order, pivot+idx] = r_mat[idx, idx+num]
                    break
    rmax_mean = rmax_raw.mean(axis=-2)
    rmin_mean = rmin_raw.mean(axis=-2)
    rmax_std = rmax_raw.std(axis=-2, ddof=1)
    rmin_std = rmin_raw.std(axis=-2, ddof=1)    
    expected_ratio_var = (rmax_mean/rmin_mean)**2 * ((rmax_std/rmax_mean)**2 + (rmin_std/rmin_mean)**2 - 2*r*rmax_std*rmin_std/(rmax_mean*rmin_mean))
    return expected_ratio_var, r

def get_cv(ref, a, b, c):
    cv = a*ref**(-b) + c
    return cv
with open("/home/md703/syu/ijv_2/shared_files/system_noise_characterization.json") as f:
    noise = json.load(f)
popt = [noise["Coefficient"]["a"],
        noise["Coefficient"]["b"],
        noise["Coefficient"]["c"]
        ]

live_max_denoise_all = np.array(live_max_denoise_all)
live_min_denoise_all = np.array(live_min_denoise_all)
live_max_denoise_all = live_max_denoise_all.reshape(-1, 
                                                    len(spec_time_denoise)//obsInv, 
                                                    live_max_denoise_all.shape[-1])
live_min_denoise_all = live_min_denoise_all.reshape(-1, 
                                                    len(spec_time_denoise)//obsInv, 
                                                    live_min_denoise_all.shape[-1])

#### Rmax and Rmin
max_intra_mean = live_max_denoise_all.mean(axis=1)
min_intra_mean = live_min_denoise_all.mean(axis=1)
max_intra_std = live_max_denoise_all.std(axis=1, ddof=1)
min_intra_std = live_min_denoise_all.std(axis=1, ddof=1)
max_intra_cv = max_intra_std / max_intra_mean
min_intra_cv = min_intra_std / min_intra_mean
max_intra_cv_inst_all = []
min_intra_cv_inst_all = []
for idx in range(max_intra_mean.shape[0]):
    
    max_mean_num = max_index_num_all[idx] / (live.shape[0] / obsInv)
    min_mean_num = min_index_num_all[idx] / (live.shape[0] / obsInv)
    max_intra_cv_inst = get_cv(max_intra_mean[idx], *popt) / np.sqrt(max_mean_num)
    min_intra_cv_inst = get_cv(min_intra_mean[idx], *popt) / np.sqrt(min_mean_num)
    max_intra_cv_inst_all.append(max_intra_cv_inst)
    min_intra_cv_inst_all.append(min_intra_cv_inst)
    
    plt.figure(figsize=(12, 4))
    lnM = plt.plot(wlpivot, max_intra_cv[idx], label="Max - total")
    lnm = plt.plot(wlpivot, min_intra_cv[idx], label="Min - total")
    plt.plot(wlpivot, max_intra_cv_inst, linestyle="--", 
            color=lnM[0].get_color(), label="Max - instrument")
    plt.plot(wlpivot, min_intra_cv_inst, linestyle="--", 
            color=lnm[0].get_color(), label="Min - instrument")
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    plt.legend()
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("CV [-]")
    plt.grid()
    plt.title(f"intra-trial variation - {idx}, Rmax and Rmin")
    plt.show()

max_intra_cv_mean = max_intra_cv.mean(axis=0)
min_intra_cv_mean = min_intra_cv.mean(axis=0)
max_intra_cv_inst_mean = np.array(max_intra_cv_inst_all).mean(axis=0)
min_intra_cv_inst_mean = np.array(min_intra_cv_inst_all).mean(axis=0)
plt.figure(figsize=(12, 4))
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
lnM = plt.plot(wlpivot, max_intra_cv_mean, label="Max - total")
lnm = plt.plot(wlpivot, min_intra_cv_mean, label="Min - total")
plt.plot(wlpivot, max_intra_cv_inst_mean, 
         linestyle="--", color=lnM[0].get_color(), label="Max - instrument")
plt.plot(wlpivot, min_intra_cv_inst_mean, 
         linestyle="--", color=lnm[0].get_color(), label="Min - instrument")
plt.legend()
plt.xlabel("Wavelength [nm]")
plt.ylabel("CV [-]")
plt.grid()
plt.title("Mean of intra-trial variation, Rmax and Rmin")
plt.show()

max_inter_mean = max_intra_mean.mean(axis=0)
max_inter_std = max_intra_mean.std(axis=0, ddof=1)
min_inter_mean = min_intra_mean.mean(axis=0)
min_inter_std = min_intra_mean.std(axis=0, ddof=1)
plt.figure(figsize=(12, 4))
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
plt.plot(wlpivot, max_inter_std/max_inter_mean, label="max")
plt.plot(wlpivot, min_inter_std/min_inter_mean, label="min")
plt.legend()
plt.xlabel("wl [nm]")
plt.ylabel("cv [-]")
plt.grid()
plt.title("Inter-trial variation, Rmax and Rmin")
plt.show()

max_total_std = live_max_denoise_all.reshape(-1, live_max_denoise_all.shape[-1]).std(axis=0, ddof=1)
min_total_std = live_min_denoise_all.reshape(-1, live_min_denoise_all.shape[-1]).std(axis=0, ddof=1)
plt.figure(figsize=(12, 4))
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
plt.plot(wlpivot, max_total_std/max_inter_mean, label="max")
plt.plot(wlpivot, min_total_std/min_inter_mean, label="min")
plt.legend()
plt.xlabel("Wavelength [nm]")
plt.ylabel("CV [-]")
plt.grid()
plt.title("Total variation, Rmax and Rmin")
plt.show()

max_intra_var_mean = np.mean(max_intra_std**2, axis=0)
min_intra_var_mean = np.mean(min_intra_std**2, axis=0)
plt.figure(figsize=(7, 4))
plt.plot(wlpivot, max_total_std**2, color=lnM[0].get_color(), 
         label="Max - total variance")
plt.plot(wlpivot, max_inter_std**2, color=lnM[0].get_color(), 
         linestyle="--", label="Max - inter-variance")
plt.plot(wlpivot, max_intra_var_mean, color=lnM[0].get_color(), 
         linestyle="-.", label="Max - intra-variance mean")
plt.legend()
plt.xlabel("Wavelength [nm]")
plt.ylabel("Variance [-]")
plt.grid()
plt.title("Variance comparison - Rmax")
plt.show()

plt.figure(figsize=(7, 4))
plt.plot(wlpivot, min_total_std**2, color=lnm[0].get_color(), 
         label="Min - total variance")
plt.plot(wlpivot, min_inter_std**2, color=lnm[0].get_color(), 
          linestyle="--", label="Min - inter-variance")
plt.plot(wlpivot, min_intra_var_mean, color=lnm[0].get_color(), 
         linestyle="-.", label="Min - intra-variance mean")
plt.legend()
plt.xlabel("Wavelength [nm]")
plt.ylabel("Variance [-]")
plt.grid()
plt.title("Variance comparison - Rmin")
plt.show()

### Rmax/Rmin
doCorr = True
exp_intra_ratio_var, r = get_expected_ratio_var(live_max_denoise_all, live_min_denoise_all, corr=doCorr)
exp_intra_ratio_mean = max_intra_mean / min_intra_mean
exp_intra_ratio_cv = np.sqrt(exp_intra_ratio_var)/exp_intra_ratio_mean
mea_ratio = live_max_denoise_all / live_min_denoise_all
mea_intra_ratio_var = np.var(mea_ratio, axis=1, ddof=1)
mea_intra_ratio_mean = mea_ratio.mean(axis=1)
mea_intra_ratio_cv = np.sqrt(mea_intra_ratio_var)/mea_intra_ratio_mean
for idx in range(max_intra_mean.shape[0]):
    
    # plt.figure(figsize=(12, 4))
    # lnM = plt.plot(wlpivot, exp_intra_ratio_var[idx], label="expected")
    # lnm = plt.plot(wlpivot, mea_intra_ratio_var[idx], label="measured")
    # plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter("{x:.2e}"))
    # plt.legend()
    # plt.xlabel("Wavelength [nm]")
    # plt.ylabel("Variance [-]")
    # plt.grid()
    # plt.title(f"Intra-trial variance - {idx}, Rmax/Rmin")
    # plt.show()    
    
    # plt.figure(figsize=(12, 4))
    # lnM = plt.plot(wlpivot, exp_intra_ratio_mean[idx], label="expected - rmax_mean/rmin_mean")
    # lnm = plt.plot(wlpivot, mea_intra_ratio_mean[idx], label="measured - mean of rmax/rmin")
    # plt.legend()
    # plt.xlabel("Wavelength [nm]")
    # plt.ylabel("Mean [-]")
    # plt.grid()
    # plt.title(f"Intra-trial Mean - {idx}, Rmax/Rmin")
    # plt.show()
    
    plt.figure(figsize=(12, 4))
    plt.plot(wlpivot, r[idx])
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Correlation coefficient [-]")
    plt.grid()
    plt.title(f"Intra-trial - {idx}, Correlation of Rmax and Rmin")
    plt.show()
    
    plt.figure(figsize=(12, 4))
    lnM = plt.plot(wlpivot, exp_intra_ratio_cv[idx], label=f"Expected - corr={doCorr}")
    lnm = plt.plot(wlpivot, mea_intra_ratio_cv[idx], label="Measured")
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    plt.legend()
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("CV [-]")
    plt.grid()
    plt.title(f"Intra-trial CV - {idx}, Rmax/Rmin")
    plt.show()

plt.figure(figsize=(12, 4))
lnM = plt.plot(wlpivot, exp_intra_ratio_cv.mean(axis=0), label="expected")
lnm = plt.plot(wlpivot, mea_intra_ratio_cv.mean(axis=0), label="measured")
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
plt.legend()
plt.xlabel("Wavelength [nm]")
plt.ylabel("CV [-]")
plt.grid()
plt.title("Intra-trial CV - mean, Rmax/Rmin")
plt.show()

max_inter_mean = max_intra_mean.mean(axis=0)
max_inter_std = max_intra_mean.std(axis=0, ddof=1)
min_inter_mean = min_intra_mean.mean(axis=0)
min_inter_std = min_intra_mean.std(axis=0, ddof=1)

exp_inter_ratio_var, r = get_expected_ratio_var(max_intra_mean, min_intra_mean, corr=doCorr)
exp_inter_ratio_mean = max_intra_mean.mean(axis=0) / min_intra_mean.mean(axis=0)
exp_inter_ratio_cv = np.sqrt(exp_inter_ratio_var) / exp_inter_ratio_mean
mea_inter_ratio_var = np.var(mea_intra_ratio_mean, axis=0, ddof=1)
mea_inter_ratio_mean = mea_intra_ratio_mean.mean(axis=0)
mea_inter_ratio_cv = np.sqrt(mea_inter_ratio_var) / mea_inter_ratio_mean

plt.figure(figsize=(12, 4))
plt.plot(wlpivot, exp_inter_ratio_mean, label="Expected")
plt.plot(wlpivot, mea_inter_ratio_mean, label="Measured")
plt.legend()
plt.xlabel("Wavelength [nm]")
plt.ylabel("Mean of Rmax/Rmin [-]")
plt.grid()
plt.title("Inter-trial mean, Rmax/Rmin")
plt.show()

plt.figure(figsize=(12, 4))
plt.plot(wlpivot, r[0])
plt.xlabel("Wavelength [nm]")
plt.ylabel("Correlation coefficient [-]")
plt.grid()
plt.title("Inter-trial, Correlation of Rmax and Rmin")
plt.show()

plt.figure(figsize=(12, 4))
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
plt.plot(wlpivot, exp_inter_ratio_cv[0], label=f"Expected - corr={doCorr}")
plt.plot(wlpivot, mea_inter_ratio_cv, label="Measured")
plt.legend()
plt.xlabel("Wavelength [nm]")
plt.ylabel("CV [-]")
plt.grid()
plt.title("Inter-trial variation, Rmax/Rmin")
plt.show()


# plot variation of raw signal - contrast
# for idx, p in enumerate(contrast_raw_all):
#     plt.scatter(np.repeat(idx, len(p)), p, marker=".", color=colorSet[0])
# means = contrast_raw_all.mean(axis=1)
# stds = contrast_raw_all.std(axis=1, ddof=1)
# mins = abs(contrast_raw_all.min(axis=1) - means)
# maxs = abs(contrast_raw_all.max(axis=1) - means)
# # plt.errorbar(np.arange(len(means)), means, yerr=[mins, maxs], 
# #               fmt="-o", capsize=4, label="contrast with range")
# plt.errorbar(np.arange(len(means)), means, yerr=stds, 
#               fmt="-", color=colorSet[0], capsize=4, label="Rmax/Rmin ± σ")
# plt.legend()
# plt.xticks(np.arange(len(means)))
# plt.xlabel("repeat index [-]")
# plt.ylabel("Rmax / Rmin  [-]")
# plt.title("stability test (5 repeats), Rmax/Rmin - raw")
# plt.show()


# plot variation of denoise signal - contrast
# for idx, p in enumerate(contrast_denoise_all):
#     plt.scatter(np.repeat(idx, len(p)), p, marker=".", color=colorSet[0])
# means = contrast_denoise_all.mean(axis=1)
# stds = contrast_denoise_all.std(axis=1, ddof=1)
# mins = abs(contrast_denoise_all.min(axis=1) - means)
# maxs = abs(contrast_denoise_all.max(axis=1) - means)
# # plt.errorbar(np.arange(len(means)), means, yerr=[mins, maxs], 
# #               fmt="-o", capsize=4, label="contrast with range")
# plt.errorbar(np.arange(len(means)), means, yerr=stds, 
#               fmt="-", color=colorSet[0], capsize=4, label="Rmax/Rmin ± σ")
# plt.legend()
# plt.xticks(np.arange(len(means)))
# plt.xlabel("repeat index [-]")
# plt.ylabel("Rmax / Rmin  [-]")
# plt.title("stability test (5 repeats), Rmax/Rmin - denoised")
# plt.show()


# # plot variation of raw signal - Δln(contrast)
# input_raw_all = np.log(contrast_raw_all)
# input_raw_all = np.diff(input_raw_all, axis=1)
# means = input_raw_all.mean(axis=1)
# stds = input_raw_all.std(axis=1, ddof=1)
# mins = abs(input_raw_all.min(axis=1) - means)
# maxs = abs(input_raw_all.max(axis=1) - means)
# plt.errorbar(np.arange(len(means)), means, yerr=[mins, maxs], 
#               fmt="-o", capsize=4, label="model input with range")
# # plt.errorbar(np.arange(len(means)), means, yerr=stds, 
# #               fmt="-o", capsize=4, label="model input ± σ")
# plt.legend()
# plt.xticks(np.arange(len(means)))
# plt.xlabel("repeat index [-]")
# plt.ylabel("Δln(Rmax/Rmin) [-]")
# plt.title("stability test (5 repeats), model input - raw")
# plt.show()


# # plot variation of denoise signal - Δln(contrast)
# input_denoise_all = np.log(contrast_denoise_all)
# input_denoise_all = np.diff(input_denoise_all, axis=1)
# means = input_denoise_all.mean(axis=1)
# stds = input_denoise_all.std(axis=1, ddof=1)
# mins = abs(input_denoise_all.min(axis=1) - means)
# maxs = abs(input_denoise_all.max(axis=1) - means)
# plt.errorbar(np.arange(len(means)), means, yerr=[mins, maxs], 
#               fmt="-o", capsize=4, label="model input with range")
# # plt.errorbar(np.arange(len(means)), means, yerr=stds, 
# #               fmt="-o", capsize=4, label="model input ± σ")
# plt.legend()
# plt.xticks(np.arange(len(means)))
# plt.xlabel("repeat index [-]")
# plt.ylabel("Δln(Rmax/Rmin) [-]")
# plt.title("stability test (5 repeats), model input - denoised")
# plt.show()


# plot contrast spectrum shape variation
# colorSet = ["blue", "red", "purple", "green", "orange"]
# for idx, contrast_wl in enumerate(contrast_wl_all):
#     for contrast in contrast_wl:
#         ln = plt.plot(targetWl, contrast, "--", color=colorSet[idx])
#     lnMean = plt.plot(targetWl, contrast_wl.mean(axis=0), "-", linewidth=5, color=colorSet[idx])
    
#     plt.legend(ln+lnMean, ["different time interval", "mean"])
#     plt.xticks(targetWl)
#     plt.xlabel("wl [nm]")
#     plt.ylabel("Rmax / Rmin  [-]")
#     plt.title(f"Rmax/Rmin spectrum variation within repeat {idx}")
#     plt.show()

# for idx, contrast_wl in enumerate(contrast_wl_all):
#     plt.plot(targetWl, contrast_wl.mean(axis=0), "-", color=colorSet[idx], label=f"exp {idx} - mean")    
# plt.legend()
# plt.xticks(targetWl)
# plt.xlabel("wl [nm]")
# plt.ylabel("Rmax / Rmin  [-]")
# plt.title("Rmax/Rmin spectrum variation between 5 repeats")
# plt.show()


# plot  Δln(contrast) spectrum shape variation
# for idx, contrast_wl in enumerate(contrast_wl_all):
#     delta = np.log(contrast_wl)
#     delta = np.diff(delta, axis=0)
#     for d in delta:
#         ln = plt.plot(targetWl, d, "--", color=colorSet[idx])
#     lnMean = plt.plot(targetWl, delta.mean(axis=0), "-", linewidth=5, color=colorSet[idx])
    
#     plt.legend(ln+lnMean, ["different time interval", "mean"])
#     plt.xticks(targetWl)
#     plt.xlabel("wl [nm]")
#     plt.ylabel("Δln(Rmax/Rmin) [-]")
#     plt.title(f"model input spectrum variation within repeat {idx}")
#     plt.show()

# for idx, contrast_wl in enumerate(contrast_wl_all):
#     delta = np.log(contrast_wl)
#     delta = np.diff(delta, axis=0)
#     plt.plot(targetWl, delta.mean(axis=0), "-", color=colorSet[idx], label=f"exp {idx} - mean") 
# plt.legend()
# plt.xticks(targetWl)
# plt.xlabel("wl [nm]")
# plt.ylabel("Δln(Rmax/Rmin) [-]")
# plt.title("model input spectrum variation between 5 repeats")
# plt.show()



