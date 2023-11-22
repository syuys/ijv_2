#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  2 16:40:27 2023

@author: md703
"""

import numpy as np
import pandas as pd
from PyEMD import EMD 
from scipy.signal import convolve
import utils
import os
import matplotlib.pyplot as plt
plt.rcParams.update({"mathtext.default": "regular"})
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["figure.dpi"] = 300


# %% function
# for signal-denoise, extract motion artifact
def get_artifact(ori_spec, showArtifact, removefromidx):
    imfs = EMD().emd(ori_spec)
    # artifact = imfs[-1] - imfs[-1].mean()
    imfs[-1] -= imfs[-1].mean()
    artifact = imfs[removefromidx:]  # get the last 3 long-period artifact (or last 2)
    
    # plot decomposition detail
    if showArtifact:
        fig, ax = plt.subplots(imfs.shape[0]+1, 1, figsize=(13, 8))
        ax[0].plot(ori_spec, 'r')
        ax[0].set_title(nameSet[idx].split('.')[0] + f", raw - ({removefromidx})")
        for n, imf in enumerate(imfs):
            ax[n+1].plot(imf, 'g')
            ax[n+1].set_title("imf " + str(n+1))        
        plt.xlabel("time [frame]")
        plt.tight_layout()
        plt.show()
    
    return artifact


# %% parameters

# select project and load wl, bg
invivo_folder = "20230706_HY_contrast_trend_upward"
nameSet = [
            "sds_10.csv", "sds_12.csv", "sds_14.csv", "sds_16.csv", "sds_18.csv", 
            "sds_20.csv", "sds_22.csv", "sds_24.csv", "sds_26.csv", "sds_28.csv", 
            "sds_30.csv", 
            "sds_33.csv", "sds_36.csv", "sds_39.csv", 
           ]
data_for_wl = nameSet[0]
data_for_bg = "background.csv"
bg_t_start = 10
bg_t_end   = 610
isCheckBG = False
normalStdTimesBG = 7
showSpikeSpecBG = False

# load raw signal and remove spike
frameStart = 30
frameEnd   = 630
selectInv = int((np.array(frameEnd) - np.array(frameStart)).mean())
isCheckRAW = False
normalStdTimesSDS = 6
showSpikeSpecSDS = False
isCheckProcessed = False

# remove motion artifact
isSaveDenoised = False
isSavePeak = False
# isSaveRmin = False
isSaveContrast = False
removefromidx = [
                -4, -4, -4, -4, -5, 
                -4, -4, -4, -4, -5, 
                -4, 
                -5, -4, -4
                 ]
split_base = 600
figSize = (13, 2.5)
isCheckCVP = False
isCheckDiffWlVarWithTime = False
DiffWlNum = 6
isCheckMeanVarWithTime = True
scattersize = 11
isShowArtifact = False
max_idx_set = []
min_idx_set = []
# slide = 23
integrationTime = 0.1
time = np.arange(0, selectInv*integrationTime, integrationTime)


# %% load wavelength, background

# wavelength
spec = pd.read_csv(os.path.join(invivo_folder, data_for_wl))
wl = np.array(spec.columns[1:].values, dtype=float) # wavelength

# background
bg = pd.read_csv(os.path.join(invivo_folder, data_for_bg))
bg = bg.iloc[bg_t_start:bg_t_end, 1:]
bg = np.array(bg)
if isCheckBG:
    for s in bg:
        plt.plot(wl, s)
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Intensity [counts]")    
    plt.title("Background - raw")
    plt.show()
bg = utils.remove_spike(wl, bg, normalStdTimesBG, showTargetSpec=showSpikeSpecBG)
if isCheckBG:
    for s in bg:
        plt.plot(wl, s)
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Intensity [counts]")    
    plt.title("Background - remove spike")
    plt.show()

# %% load experiment
signal_raw = np.empty((len(nameSet), selectInv, len(wl)), dtype=float)
for nameIdx, name in enumerate(nameSet):
    title = name.split(".")[0]
    spec = pd.read_csv(os.path.join(invivo_folder, name))
    print(f"{title}: {spec.shape} → ", end="")
    spec = spec.iloc[frameStart:frameEnd, 1:]   # drop first column (time stamp)
    # spec = spec.iloc[frameStart[nameIdx]:frameEnd[nameIdx], 1:]   # drop first column (time stamp)
    spec = np.array(spec)
    print(spec.shape, end=", ")
    
    # plot raw signal
    if isCheckRAW:
        for s in spec:
            plt.plot(wl, s)
        plt.xlabel("Wavelength [nm]")
        plt.ylabel("Intensity [counts]")    
        plt.title(f"{title} - raw")
        plt.show()
    
    # subtract background and detect spike
    spec -= bg.mean(axis=0)    
    spec = utils.remove_spike(wl, spec, normalStdTimesSDS, showTargetSpec=showSpikeSpecSDS)
    
    # save signal with removal of background and spike
    signal_raw[nameIdx] = spec
    
    # plot processed signal
    if isCheckProcessed:
        for s in spec:
            plt.plot(wl, s)
        plt.xlabel("Wavelength [nm]")
        plt.ylabel("Intensity [counts]")        
        plt.title(f"{title} - remove spike and background")
        plt.show()


# %% extract and remove motion artifact, extract Rmax and Rmin
subject = invivo_folder.split("_")[1]

signal_denoised = np.empty((signal_raw.shape[0], signal_raw.shape[1], len(wl)), dtype=float)
for idx, spec_raw in enumerate(signal_raw):    
    
    # CVP
    if isCheckCVP:
        cvp = spec_raw.sum(axis=1)
        cvp = 1 / cvp
        plt.figure(figsize=figSize)    
        plt.plot(time, cvp)
        plt.grid()
        plt.xlabel(f"Time [s], integration time = {integrationTime} s")
        plt.ylabel("1 / Sum(intensity)  [-]")
        plt.title(f"Trial {nameSet[idx].split('.')[0]} - CVP waveform of raw signal")
        plt.show()    
    
    # detect motion artifact and remove    
    spec_raw_time = spec_raw.mean(axis=1)  # do average w.r.t wavelength    
    artifact = get_artifact(spec_raw_time, showArtifact=isShowArtifact, removefromidx=removefromidx[idx])
    spec_denoised = spec_raw.copy()
    for art in artifact:
        spec_denoised -= art.reshape(-1, 1)
    signal_denoised[idx] = spec_denoised
    
    if isCheckDiffWlVarWithTime:
        plt.figure(figsize=figSize)
        rndSet = np.random.choice(len(wl)//3, size=DiffWlNum, replace=False)
        rndSet.sort()
        for rnd in rndSet:
            plt.plot(time, spec_denoised[:, rnd], label=f"{int(np.around(wl[rnd], 0))} nm")
        plt.grid()
        plt.legend()
        plt.xlabel(f"Time [s], integration time = {integrationTime} s")
        plt.ylabel("Intensity [counts]")
        plt.title(f"Trial {nameSet[idx].split('.')[0]} - pattern of different wavelength")
        plt.show()
    
    # detect peak 
    spec_denoised_time = spec_denoised.mean(axis=1)  # do average w.r.t wavelength
    max_idx, min_idx = utils.get_peak_final(spec_denoised_time)
    max_idx_set.append(max_idx)
    min_idx_set.append(min_idx)
    
    # save denoised rmax, rmin
    if isSavePeak:
        rmax_denoised_df = pd.DataFrame(spec_denoised[max_idx], columns=wl)
        rmin_denoised_df = pd.DataFrame(spec_denoised[min_idx], columns=wl)
        title = nameSet[idx].split(".")[0]
        rmax_denoised_df.to_csv(os.path.join(invivo_folder, 
                                             title+"_denoised_rmax"+".csv"), 
                                    index=False)
        rmin_denoised_df.to_csv(os.path.join(invivo_folder, 
                                             title+"_denoised_rmin"+".csv"), 
                                    index=False)
    
    # save denoised spec
    if isSaveDenoised:
        spec_denoised_df = pd.DataFrame(spec_denoised, columns=wl)
        title = nameSet[idx].split(".")[0]
        spec_denoised_df.to_csv(os.path.join(invivo_folder, title+"_denoised"+".csv"), 
                                    index=False)
    
    # plot raw, denoised comparison
    if isCheckMeanVarWithTime:
        split_num = np.ceil(len(spec_denoised_time)/split_base).astype(int)
        for split_idx in range(split_num):
            split_s = split_base*split_idx
            split_e = split_base*(split_idx+1)
            max_idx_local = max_idx[(max_idx>=split_s) & (max_idx<split_e)]
            min_idx_local = min_idx[(min_idx>=split_s) & (min_idx<split_e)]
            
            # raw
            plt.figure(figsize=figSize)
            plt.plot(time[split_s:split_e], spec_raw_time[split_s:split_e])
            plt.scatter(max_idx_local*integrationTime, 
                        spec_raw_time[max_idx_local], s=scattersize, 
                        color="red", label="Max")
            plt.scatter(min_idx_local*integrationTime, 
                        spec_raw_time[min_idx_local], s=scattersize, 
                        color="tab:orange", label="Min")
            plt.title(f"{subject}, trial {nameSet[idx].split('.')[0]} - raw")
            plt.legend()
            plt.grid()
            plt.xlabel(f"Time [s], integration time = {integrationTime} s")
            plt.ylabel("Mean of spectrum [counts]")
            ylim = plt.ylim()
            plt.show()    
        
            # denoised
            plt.figure(figsize=figSize)
            plt.plot(time[split_s:split_e], spec_denoised_time[split_s:split_e])
            plt.scatter(max_idx_local*integrationTime, 
                        spec_denoised_time[max_idx_local], s=scattersize, 
                        color="red", label="Max")
            plt.scatter(min_idx_local*integrationTime, 
                        spec_denoised_time[min_idx_local], s=scattersize, 
                        color="tab:orange", label="Min")
            plt.title(f"{subject}, trial {nameSet[idx].split('.')[0]} - denosied ({removefromidx[idx]})")
            plt.legend()
            plt.grid()
            plt.ylim(ylim)
            plt.xlabel(f"Time [s], integration time = {integrationTime} s")
            plt.ylabel("Mean of spectrum [counts]")
            plt.show()


# %% calculate rmax/rmin
import matplotlib.ticker as mtick
wltarget = [730, 760, 780, 810, 850]
obsInv = selectInv
invNum = selectInv // obsInv
rmaxSet = np.empty((signal_denoised.shape[0], invNum, len(wl)))
rminSet = np.empty((signal_denoised.shape[0], invNum, len(wl)))
# rmaxwlSet = np.empty((len(wltarget), signal_denoised.shape[0], invNum))
# rminwlSet = np.empty((len(wltarget), signal_denoised.shape[0], invNum))
for idx, spec in enumerate(signal_denoised):
    rmax = np.empty((invNum, len(wl)))
    rmin = np.empty((invNum, len(wl)))
    max_idx = max_idx_set[idx]
    min_idx = min_idx_set[idx]
    for intra_idx in range(invNum):
        max_idx_local = max_idx[(max_idx >= intra_idx*obsInv) & (max_idx < (intra_idx+1)*obsInv)]
        min_idx_local = min_idx[(min_idx >= intra_idx*obsInv) & (min_idx < (intra_idx+1)*obsInv)]
        rmax[intra_idx] = spec[max_idx_local].mean(axis=0)
        rmin[intra_idx] = spec[min_idx_local].mean(axis=0)
    rmaxSet[idx] = rmax
    rminSet[idx] = rmin

# rmaxbyrmin = rmaxSet / rminSet
# rmaxbyrminMean = rmaxbyrmin.mean(axis=1)

wlInvNumSet = [0, 2, 8, 32]
sdsSet = np.array([float(name.split(".")[0].split("_")[1]) for name in nameSet])

# print(f"rmaxSet shape: {rmaxSet.shape}")
rmaxSet, rminSet = rmaxSet.mean(axis=1), rminSet.mean(axis=1)
rmaxSetMean, rminSetMean = rmaxSet.mean(axis=1), rminSet.mean(axis=1)
contrastMean = rmaxSetMean/rminSetMean
for wlInvNum in wlInvNumSet:
    if wlInvNum != 0:
        wlNumInInv = len(wl)//wlInvNum
        start_pivot = -256
        for wlInvIdx in range(wlInvNum):
            start = wlNumInInv*wlInvIdx
            end = wlNumInInv*(wlInvIdx+1)
            if (start - start_pivot >= 256):
                rmaxbyrmin = rmaxSet[:, start:end].mean(axis=1) / rminSet[:, start:end].mean(axis=1)
                plt.plot(sdsSet, rmaxbyrmin-1, label=f"{round(wl[start])}nm - {round(wl[end-1])}nm",
                         color=utils.colorFader("blue", "red", wlInvIdx/(wlInvNum-1)))
                start_pivot = start
            else:
                continue
    plt.plot(sdsSet, contrastMean-1, label="All", color="gray")
    plt.legend()
    plt.xticks(sdsSet)
    plt.xlabel("SDS [mm]")
    plt.ylabel("Rmax/Rmin - 1  [-]")
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    plt.grid()
    plt.title(f"{subject} - Contrast mean v.s. SDS")
    plt.show()
# if isSaveRmin:
#     np.save(file=os.path.join(invivo_folder, f"{subject}_rmin"), 
#             arr=rminSet)
if isSaveContrast:
    contrast_df = np.concatenate((sdsSet[:, None], contrastMean[:, None]), axis=1)
    contrast_df = pd.DataFrame(contrast_df, columns=["SDS [mm]", "Rmax / Rmin  [-]"])
    contrast_df.to_csv(os.path.join(invivo_folder, subject+"_contrast_trend.csv"), 
                       index=False)

# for wlidx, wlvalue in enumerate(wltarget):
#     plt.plot(sdsSet, rmaxbyrminWlSnr[:, wlidx], 
#              color=utils.colorFader("blue", "red", wlidx/(len(wltarget)-1)), 
#              marker=".", linestyle="-", 
#              label=f"{wlvalue} nm")
# plt.plot(sdsSet, rmaxbyrminSnr.mean(axis=1), marker=".", linestyle="-", color="gray", label="All wl mean")
# plt.legend()
# plt.xticks(sdsSet)
# plt.xlabel("SDS [mm]")
# plt.ylabel("(Rmax/Rmin)_mean / (Rmax/Rmin)_std  [-]")
# plt.grid()
# plt.title("CNR v.s. SDS")
# plt.show()

# for wlidx, wlvalue in enumerate(wltarget):
#     plt.plot(sdsSet, (1/rmaxbyrminWlSnr)[:, wlidx], 
#              color=utils.colorFader("blue", "red", wlidx/(len(wltarget)-1)), 
#              marker=".", linestyle="-", 
#              label=f"{wlvalue} nm")
# plt.plot(sdsSet, (1/rmaxbyrminSnr).mean(axis=1), marker=".", linestyle="-", color="gray", label="All wl mean")
# plt.legend()
# plt.xticks(sdsSet)
# plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
# plt.xlabel("SDS [mm]")
# plt.ylabel("(Rmax/Rmin)_std / (Rmax/Rmin)_mean  [-]")
# plt.grid()
# plt.title("CV v.s. SDS")
# plt.show()