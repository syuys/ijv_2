#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  2 16:40:27 2023

@author: md703
"""

import numpy as np
import pandas as pd
from scipy import signal
import utils
import os
import json
import matplotlib as mpl
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science', 'grid'])
# plt.rcParams.update({"mathtext.default": "regular"})
# plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["figure.dpi"] = 300


# %% function
# estimate Var(Rmax/Rmin)
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


# get estimated instrumental noise
def get_cv(ref, a, b, c):
    cv = a*ref**(-b) + c
    return cv


# %% parameters
# 20230601_HW_contrast_trend_and_holder_stability, 20230507_HW_holder_stability_test
# 20230812_differentSDS_detectorCV
invivo_folder = "20230812_differentSDS_detectorCV"
nameSet = [
            # "sds_10_0_denoised.csv", "sds_10_1_denoised.csv", 
            # "sds_10_2_denoised.csv", "sds_10_3_denoised.csv", 
            # "sds_10_4_denoised.csv", "sds_10_5_denoised.csv",
            "sds_20_0_denoised.csv", "sds_20_1_denoised.csv", 
            "sds_20_2_denoised.csv", "sds_20_3_denoised.csv", 
            "sds_20_4_denoised.csv", "sds_20_5_denoised.csv",
            # "sds_30_0_denoised.csv", "sds_30_1_denoised.csv", 
            # "sds_30_2_denoised.csv", "sds_30_3_denoised.csv", 
            # "sds_30_4_denoised.csv", "sds_30_5_denoised.csv",
            # "det_1_denoised.csv",
            # "det_2_denoised.csv",
            # "det_3_denoised.csv",
            # "det_4_denoised.csv",
            # "det_5_denoised.csv",
           ]
# dataTimeLen = 60  # second [s]
figSize = (12, 4)
inst_noise_charac_file = "shared_files/system_noise_characterization.json"
wl_window = 23
# obsInv = 30  # for observing Rmax/Rmin variation within one experiment
peak_num = 100  # initial peak num
max_idx_set = []
min_idx_set = []
signal_denoised_mov = []


# %% Analyze the signal after doing denoise (remove motion artifact w.r.t time) and moving-average (w.r.t wavelength)

### retrieve wl, timeNum, rmax index, rmin index
pivot = (wl_window-1) // 2
tmp = pd.read_csv(os.path.join(invivo_folder, nameSet[0]))
wlpivot = tmp.columns.values.astype(float)[pivot: -pivot]
timeNum = tmp.shape[0]
signal_denoised_mov = np.empty((len(nameSet), timeNum, len(wlpivot)))
for idx, name in enumerate(nameSet):
    # read data
    spec_denoised_mov = pd.read_csv(os.path.join(invivo_folder, name))
    spec_denoised_mov = np.array(spec_denoised_mov)
    
    # detect peak and show
    spec_denoised_mov_time = spec_denoised_mov.mean(axis=1)
    max_idx, min_idx = utils.get_peak_final(spec_denoised_mov_time)
    max_idx_set.append(max_idx)
    min_idx_set.append(min_idx)
    
    # plt.figure(figsize=(13, 2.5))
    # plt.plot(np.arange(0, dataTimeLen, 0.1), spec_denoised_mov_time)
    # plt.scatter(max_idx*0.1, 
    #             spec_denoised_mov_time[max_idx], s=11, 
    #             color="red", label="Max")
    # plt.scatter(min_idx*0.1, 
    #             spec_denoised_mov_time[min_idx], s=11, 
    #             color="tab:orange", label="Min")
    # plt.title(f"Trial {idx}")
    # plt.legend()
    # plt.grid()
    # plt.xlabel("Time [s], integration time = 0.1 s")
    # plt.ylabel("Mean of spectrum [counts]")
    # plt.show()
    
    # do moving average w.r.t wl
    spec_denoised_mov = signal.convolve(spec_denoised_mov, 
                                        np.ones((1, wl_window))/wl_window, 
                                        mode='valid')
    signal_denoised_mov[idx] = spec_denoised_mov

### check mean rmax, rmin of each trial
# for idx, spec in enumerate(signal_denoised_mov):
#     rmax = spec[max_idx_set[idx]].mean(axis=0)
#     rmin = spec[min_idx_set[idx]].mean(axis=0)
#     plt.figure(figsize=figSize)
#     plt.plot(wlpivot, rmax, label="Max")
#     plt.plot(wlpivot, rmin, label="Min")
#     plt.grid()
#     plt.legend()
#     plt.xlabel("Wavelength [nm]")
#     plt.ylabel("Mean intensity [counts]")
#     plt.title(f"Trial - {idx}")
#     plt.show()


### extract detailed rmax, rmin
for max_idx, min_idx in zip(max_idx_set, min_idx_set):
    minpeaknum = min(len(max_idx), len(min_idx))
    if minpeaknum < peak_num:
        peak_num = minpeaknum
# invNum = timeNum//obsInv
rmax_denoised_mov = np.empty((len(nameSet), peak_num, len(wlpivot)), dtype=float)
rmin_denoised_mov = np.empty((len(nameSet), peak_num, len(wlpivot)), dtype=float)
for inter_idx, spec in enumerate(signal_denoised_mov):
    rmax_denoised_mov[inter_idx] = spec[max_idx_set[inter_idx][:peak_num]]
    rmin_denoised_mov[inter_idx] = spec[min_idx_set[inter_idx][:peak_num]]
    # for intra_idx in range(rmax_denoised_mov.shape[1]):
    #     max_idx_local = max_idx[(max_idx >= intra_idx*obsInv) & (max_idx < (intra_idx+1)*obsInv)]
    #     min_idx_local = min_idx[(min_idx >= intra_idx*obsInv) & (min_idx < (intra_idx+1)*obsInv)]
    #     rmax_denoised_mov[inter_idx][intra_idx] = spec[max_idx_local].mean(axis=0)
    #     rmin_denoised_mov[inter_idx][intra_idx] = spec[min_idx_local].mean(axis=0)


### check rmax, rmin variation
# intra-trial
with open(inst_noise_charac_file) as f:
    inst_noise = json.load(f)
popt = [inst_noise["Coefficient"]["a"],
        inst_noise["Coefficient"]["b"],
        inst_noise["Coefficient"]["c"]
        ]

rmax_intra_std = rmax_denoised_mov.std(axis=1, ddof=1)
rmin_intra_std = rmin_denoised_mov.std(axis=1, ddof=1)
rmax_intra_mean = rmax_denoised_mov.mean(axis=1)
rmin_intra_mean = rmin_denoised_mov.mean(axis=1)
rmax_intra_cv = rmax_intra_std / rmax_intra_mean
rmin_intra_cv = rmin_intra_std / rmin_intra_mean
rmax_intra_num = np.array([len(max_idx) for max_idx in max_idx_set])
rmin_intra_num = np.array([len(min_idx) for min_idx in min_idx_set])
rmax_intra_cv_inst = get_cv(rmax_intra_mean, *popt)  # / np.sqrt(rmax_intra_num/invNum)[:, None]
rmin_intra_cv_inst = get_cv(rmin_intra_mean, *popt)  # / np.sqrt(rmin_intra_num/invNum)[:, None]

rmax_inter_std  = rmax_intra_mean.std(axis=0, ddof=1)
rmax_inter_mean = rmax_intra_mean.mean(axis=0)
rmin_inter_std  = rmin_intra_mean.std(axis=0, ddof=1)
rmin_inter_mean = rmin_intra_mean.mean(axis=0)
rmax_inter_cv = rmax_inter_std / rmax_inter_mean
rmin_inter_cv = rmin_inter_std / rmin_inter_mean
rmax_total_std = rmax_denoised_mov.reshape(-1, len(wlpivot)).std(axis=0, ddof=1)
rmin_total_std = rmin_denoised_mov.reshape(-1, len(wlpivot)).std(axis=0, ddof=1)
rmax_total_cv  = rmax_total_std / rmax_inter_mean
rmin_total_cv  = rmin_total_std / rmin_inter_mean

rmax_intra_var_mean = np.mean(rmax_intra_std**2, axis=0)
rmin_intra_var_mean = np.mean(rmin_intra_std**2, axis=0)
rmax_inter_var = rmax_inter_std**2
rmin_inter_var = rmin_inter_std**2
rmax_total_var = rmax_total_std**2
rmin_total_var = rmin_total_std**2

fig, ax = plt.subplots(1, 2, figsize=(8, 3))
for inter_idx, (intra_rmax, intra_rmin) in enumerate(zip(rmax_denoised_mov, rmin_denoised_mov)):
    
    # intra_rmax_time = intra_rmax.mean(axis=-1)
    # intra_rmin_time = intra_rmin.mean(axis=-1)    
    # plt.figure(figsize=(6, 1.5))
    lm = ax[0].plot(wlpivot, intra_rmax.mean(axis=0), label=f"Trial - {inter_idx}",
                    # color=utils.colorFader(c1="blue", c2="red", mix=inter_idx/(rmax_denoised_mov.shape[0]-1))
                    )
    ax[1].plot(wlpivot, intra_rmin.mean(axis=0), label=f"Trial - {inter_idx}",
                  color=lm[0].get_color())
    # plt.plot(intra_rmax_time, marker="o", linestyle="-", label="Max")
    # plt.plot(intra_rmin_time, marker="o", linestyle="-", label="Min")
    # plt.grid(visible=False)
    # plt.legend()
    # plt.xlabel("Index of time interval [-]")
    # plt.ylabel("Mean intensity [counts]")
    # plt.title(f"Trial - {inter_idx}")
    # plt.show()
    
    # r = np.corrcoef(intra_rmin_time, intra_rmax_time)[0, 1]
    # plt.plot(intra_rmin_time, intra_rmax_time, "o")
    # plt.grid()
    # plt.xlabel("Intensity of Rmin [counts]")
    # plt.ylabel("Intensity of Rmax [counts]")
    # plt.title(f"Trial - {inter_idx}, r = {np.around(r, 2)}")
    # plt.show()
    
    # plt.figure(figsize=figSize)
    # lnM = plt.plot(wlpivot, rmax_intra_cv[inter_idx], label="Max - total")
    # lnm = plt.plot(wlpivot, rmin_intra_cv[inter_idx], label="Min - total")
    # plt.plot(wlpivot, rmax_intra_cv_inst[inter_idx], linestyle="--", 
    #         color=lnM[0].get_color(), label="Max - instrument")
    # plt.plot(wlpivot, rmin_intra_cv_inst[inter_idx], linestyle="--", 
    #         color=lnm[0].get_color(), label="Min - instrument")
    # plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    # plt.legend()
    # plt.xlabel("Wavelength [nm]")
    # plt.ylabel("CV [-]")
    # plt.grid()
    # plt.title(f"Intra-trial variation - {inter_idx}, Rmax and Rmin")
    # plt.show()
ax[0].grid(visible=False)
ax[1].grid(visible=False)
ax[0].set_xlabel("Wavelength [nm]")
ax[1].set_xlabel("Wavelength [nm]")
ax[0].set_ylabel("Intensity [counts]")
# ax[1].set_ylabel("Intensity [counts]")
# ax[1].legend(edgecolor="black", 
#               fontsize="small",
#               loc='upper right', 
#               bbox_to_anchor=(1.35, 1.03))
handles, labels = ax[0].get_legend_handles_labels()
fig.legend(handles, labels, edgecolor="black", 
           loc='lower center', bbox_to_anchor=(0.51, -0.06),
           ncol=6, fontsize="small")
ax[0].set_title("Rmax (6 repeats)")
ax[1].set_title("Rmin (6 repeats)")
plt.tight_layout()
plt.show()

# different wavelength corrcoeff for rmax, rmin (probe placement)
def add(s, n):
    c = 0
    for i in range(n):
        c += (s-i)
    return c

wlcorr = np.linspace(wlpivot[0], wlpivot[-1], 30)
rmaxcorr = np.empty((rmax_intra_mean.shape[0], wlcorr.shape[0]))
rmincorr = np.empty((rmin_intra_mean.shape[0], wlcorr.shape[0]))
for idx, (rmax, rmin) in enumerate(zip(rmax_intra_mean, rmin_intra_mean)):
    rmaxcorr[idx] = np.interp(wlcorr, wlpivot, rmax_intra_mean[idx])
    rmincorr[idx] = np.interp(wlcorr, wlpivot, rmin_intra_mean[idx])
# rmaxcorrcoef = np.corrcoef(rmaxcorr.T)[np.triu_indices(rmaxcorr.T.shape[0], 1)]
rmaxcorrcoef = np.corrcoef(rmaxcorr.T)
# rmincorrcoef = np.corrcoef(rmincorr.T)[np.triu_indices(rmincorr.T.shape[0], 1)]
rmincorrcoef = np.corrcoef(rmincorr.T)

rowidx = np.triu_indices(rmaxcorr.T.shape[0], 1)[0]
colidx = np.triu_indices(rmaxcorr.T.shape[0], 1)[1]
rmaxcorrcoef_wl = []
rmincorrcoef_wl = []
for wlidx in range(len(wlcorr)):
    match = (wlidx == rowidx) | (wlidx == colidx)
    rmaxcorrcoef_wl.append(np.corrcoef(rmaxcorr.T)[(rowidx[match], colidx[match])].mean())
    rmincorrcoef_wl.append(np.corrcoef(rmincorr.T)[(rowidx[match], colidx[match])].mean())

fig, ax = plt.subplots(1, 2, figsize=(8, 3))
c = 0
for wlidx, (rmaxcorrcoef_row, rmincorrcoef_row) in enumerate(zip(rmaxcorrcoef, rmincorrcoef)):
    indexSet = list(range(c, c+rmaxcorrcoef_row[wlidx+1:].shape[0]))
    # for idx, (maxcoef, mincoef) in enumerate(zip(rmaxcorrcoef_row[wlidx+1:], rmincorrcoef_row[wlidx+1:])):
    ax[0].plot(indexSet, rmaxcorrcoef_row[wlidx+1:], ".", markersize=2,
                # color=utils.colorFader(c1="blue", c2="red", mix=wlidx/(rmaxcorrcoef_row.shape[0]-1))
                )
    ax[1].plot(indexSet, rmincorrcoef_row[wlidx+1:], ".", markersize=2,
                # color=utils.colorFader(c1="blue", c2="red", mix=wlidx/(rmincorrcoef_row.shape[0]-1))
                )
    c += rmaxcorrcoef_row[wlidx+1:].shape[0]
# ax[0].plot(rmaxcorrcoef, ".", markersize=2)
# ax[1].plot(rmincorrcoef, ".", markersize=2)
# piv = 100
# ax[0].axvline(x=add(wlpivot.shape[0]-1, piv), 
#               # ymin=ax[0].set_ylim()[0], ymax=ax[0].set_ylim()[1],
#               color="orange",
#               label=f"wl = {wlpivot[piv]} nm")
# ax[1].axvline(x=add(wlpivot.shape[0]-1, piv), 
#               # ymin=ax[1].set_ylim()[0], ymax=ax[1].set_ylim()[1],
#               color="orange",
#               label=f"wl = {wlpivot[piv]} nm")
# piv = 300
# ax[0].axvline(x=add(wlpivot.shape[0]-1, piv), 
#               # ymin=ax[0].set_ylim()[0], ymax=ax[0].set_ylim()[1],
#               color="red",
#               label=f"wl = {wlpivot[piv]} nm")
# ax[1].axvline(x=add(wlpivot.shape[0]-1, piv), 
#               # ymin=ax[1].set_ylim()[0], ymax=ax[1].set_ylim()[1],
#               color="red",
#               label=f"wl = {wlpivot[piv]} nm")
# ax[1].legend(edgecolor="black", 
#               fontsize="small",
#               loc='upper right', 
#               bbox_to_anchor=(1.6, 1.03))
ax[0].grid(visible=False)
ax[1].grid(visible=False)
ax[0].set_xlabel("Index of different wavelength combination [-]")
ax[1].set_xlabel("Index of different wavelength combination [-]")
ax[0].set_ylabel("Correlation coefficient [-]")
ax[0].set_title("Rmax")
ax[1].set_title("Rmin")
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(1, 2, figsize=(8, 3))
ax[0].plot(wlcorr, rmaxcorrcoef_wl, marker=".")
ax[1].plot(wlcorr, rmincorrcoef_wl, marker=".")
ax[0].grid(visible=False)
ax[1].grid(visible=False)
ax[0].set_xlabel("Wavelength [nm]")
ax[1].set_xlabel("Wavelength [nm]")
ax[0].set_ylabel("Average correlation coefficient [-]")
ax[0].set_title("Rmax")
ax[1].set_title("Rmin")
plt.tight_layout()
plt.show()

# mean of intra
# plt.figure(figsize=figSize)
# plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
# lnM = plt.plot(wlpivot, rmax_intra_cv.mean(axis=0), label="Max - intra")
# lnm = plt.plot(wlpivot, rmin_intra_cv.mean(axis=0), label="Min - intra")
# plt.plot(wlpivot, rmax_intra_cv_inst.mean(axis=0), 
#          linestyle="--", color=lnM[0].get_color(), label="Max - instrument")
# plt.plot(wlpivot, rmin_intra_cv_inst.mean(axis=0), 
#          linestyle="--", color=lnm[0].get_color(), label="Min - instrument")
# plt.legend()
# plt.xlabel("Wavelength [nm]")
# plt.ylabel("CV [-]")
# plt.grid()
# plt.title("Mean of intra-trial variation, Rmax and Rmin")
# plt.show()

# # compare rmax variance (measured total, inter, intra)
# plt.figure(figsize=figSize)
# plt.plot(wlpivot, rmax_total_var, linestyle="-", color=lnM[0].get_color(), 
#          label="Total")
# plt.plot(wlpivot, rmax_inter_var, linestyle="--", color=lnM[0].get_color(), 
#          label="Inter")
# plt.plot(wlpivot, rmax_intra_var_mean, linestyle="-.", color=lnM[0].get_color(), 
#          label="Intra")
# plt.legend()
# plt.xlabel("Wavelength [nm]")
# plt.ylabel("Variance [-]")
# plt.grid()
# plt.title("Comparison of variance, Rmax")
# plt.show()

# # compare rmin variance (measured total, inter, intra)
# plt.figure(figsize=figSize)
# plt.plot(wlpivot, rmin_total_var, linestyle="-", color=lnm[0].get_color(), 
#          label="Total")
# plt.plot(wlpivot, rmin_inter_var, linestyle="--", color=lnm[0].get_color(), 
#          label="Inter")
# plt.plot(wlpivot, rmin_intra_var_mean, linestyle="-.", color=lnm[0].get_color(), 
#          label="Intra")
# plt.legend()
# plt.xlabel("Wavelength [nm]")
# plt.ylabel("Variance [-]")
# plt.grid()
# plt.title("Comparison of Rmin variance, Rmin")
# plt.show()

# # compare rmax variance (measured total, intra and estimated inter)
# plt.figure(figsize=figSize)
# plt.plot(wlpivot, rmax_total_var, linestyle="-", color=lnM[0].get_color(), 
#          label="Total")
# plt.plot(wlpivot, rmax_total_var-rmax_intra_var_mean, linestyle="--", color=lnM[0].get_color(), 
#          label="Inter (estimated)")
# plt.plot(wlpivot, rmax_intra_var_mean, linestyle="-.", color=lnM[0].get_color(), 
#          label="Intra")
# plt.legend()
# plt.xlabel("Wavelength [nm]")
# plt.ylabel("Variance [-]")
# plt.grid()
# plt.title("Comparison of variance, Rmax")
# plt.show()

# compare rmin variance (measured total, intra and estimated inter)
# plt.figure(figsize=figSize)
# plt.plot(wlpivot, rmin_total_var, linestyle="-", 
#          # color=lnm[0].get_color(), 
#          label="Total")
# plt.plot(wlpivot, rmin_total_var-rmin_intra_var_mean, 
#          # linestyle="--", 
#          # color=lnm[0].get_color(), 
#          label="Inter (estimated)")
# plt.plot(wlpivot, rmin_intra_var_mean, 
#          # linestyle="-.", 
#          # color=lnm[0].get_color(), 
#          label="Intra")
# plt.legend()
# plt.xlabel("Wavelength [nm]")
# plt.ylabel("Variance [-]")
# plt.grid()
# plt.title("Comparison of variance, Rmin")
# plt.show()

# compare rmax cv (measured total, intra and estimated inter)
# plt.figure(figsize=(8.4, 2))
# plt.plot(wlpivot, rmax_total_cv, linestyle="-", 
#          # color=lnM[0].get_color(), 
#           label="Total")
# plt.plot(wlpivot, np.sqrt(rmax_total_var-rmax_intra_var_mean) / rmax_inter_mean, 
#           linestyle="--", 
#           # color=lnM[0].get_color(), 
#           label="Inter (estimated)")
# plt.plot(wlpivot, np.sqrt(rmax_intra_var_mean) / rmax_inter_mean, linestyle="-.", 
#           # color=lnM[0].get_color(), 
#           label="Intra")
# plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
# plt.legend()
# plt.xlabel("Wavelength [nm]")
# plt.ylabel("CV [-]")
# plt.grid()
# plt.title("Comparison of CV, Rmax")
# plt.show()

# compare rmin cv (measured total, intra and estimated inter)
# plt.figure(figsize=(8.4, 2))
# plt.plot(wlpivot, rmin_total_cv, linestyle="-", 
#           # color=lnm[0].get_color(), 
#           label="Total")
# plt.plot(wlpivot, np.sqrt(rmin_total_var-rmin_intra_var_mean) / rmin_inter_mean, 
#           # linestyle="--", 
#           # color=lnm[0].get_color(), 
#           label="Inter (estimated)")
# plt.plot(wlpivot, np.sqrt(rmin_intra_var_mean) / rmin_inter_mean, 
#           # linestyle="-.", 
#           # color=lnm[0].get_color(), 
#           label="Intra")
# plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
# plt.legend()
# plt.xlabel("Wavelength [nm]")
# plt.ylabel("CV [-]")
# plt.grid()
# plt.title("Comparison of CV, Rmin")
# plt.show()

# compare rmax and rmin estimated-inter cv
plt.figure(figsize=(6, 1.5))
plt.plot(wlpivot, np.sqrt(rmax_total_var-rmax_intra_var_mean) / rmax_inter_mean, 
         label="Rmax")
plt.plot(wlpivot, np.sqrt(rmin_total_var-rmin_intra_var_mean) / rmin_inter_mean, 
         label="Rmin")
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
plt.legend(edgecolor="black")
plt.xlabel("Wavelength [nm]")
plt.ylabel("CV [-]")
plt.grid(visible=False)
plt.title(f"Comparison of Rmax and Rmin - probe placement, SDS = {nameSet[0].split('_')[1]} mm")
plt.show()

# # compare intra cv - rmax (through mean variance and through mean cv)
# plt.figure(figsize=figSize)
# plt.plot(wlpivot, rmax_intra_cv.mean(axis=0), label="Mean intra")
# plt.plot(wlpivot, np.sqrt(rmax_intra_var_mean) / rmax_inter_mean, 
#          label="Mean variance / Total ref mean")
# plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
# plt.legend()
# plt.xlabel("Wavelength [nm]")
# plt.ylabel("CV [-]")
# plt.grid()
# plt.title("Comparison of Intra CV, Rmax")
# plt.show()

# # compare intra cv - rmin (through mean variance and through mean cv)
# plt.figure(figsize=figSize)
# plt.plot(wlpivot, rmin_intra_cv.mean(axis=0), label="Mean intra")
# plt.plot(wlpivot, np.sqrt(rmin_intra_var_mean) / rmin_inter_mean, 
#          label="Mean variance / Total ref mean")
# plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
# plt.legend()
# plt.xlabel("Wavelength [nm]")
# plt.ylabel("CV [-]")
# plt.grid()
# plt.title("Comparison of Intra CV, Rmin")
# plt.show()


# %% check metric variation (rmax/rmin, Δln(rmax/rmin), ... etc)
metricName = "Rmax/Rmin"  # Δln(Rmax/Rmin)
doCorr = True
exp_metric_intra_var, r_intra = get_expected_ratio_var(rmax_denoised_mov, rmin_denoised_mov, corr=doCorr)
exp_metric_intra_mean = rmax_intra_mean / rmin_intra_mean
exp_metric_intra_cv = np.sqrt(exp_metric_intra_var) / exp_metric_intra_mean

exp_metric_inter_var, r_inter = get_expected_ratio_var(rmax_intra_mean, rmin_intra_mean, corr=doCorr)
exp_metric_inter_mean = rmax_inter_mean / rmin_inter_mean
exp_metric_inter_cv = np.sqrt(exp_metric_inter_var) / exp_metric_inter_mean

exp_metric_total_var, r_total = get_expected_ratio_var(rmax_denoised_mov.reshape(-1, len(wlpivot)), 
                                                        rmin_denoised_mov.reshape(-1, len(wlpivot)), 
                                                        corr=doCorr)
exp_metric_total_cv = np.sqrt(exp_metric_total_var) / exp_metric_inter_mean

if metricName == "Rmax/Rmin":
    metric_denoised_mov = rmax_denoised_mov / rmin_denoised_mov
    
elif metricName == "Δln(Rmax/Rmin)":
    shift = 1
    metric_denoised_mov = rmax_denoised_mov / rmin_denoised_mov
    metric_denoised_mov = np.diff(np.log(metric_denoised_mov), axis=1) + shift  # +1 is to shift the mean from 0 to 1
    
    exp_metric_intra_var /= (rmax_intra_mean/rmin_intra_mean)**2
    exp_metric_intra_var *= 2
    exp_metric_intra_mean = shift  # EX(ln(Rmax/Rmin)) - EX(ln(Rmax/Rmin)) = 0
    exp_metric_intra_cv = np.sqrt(exp_metric_intra_var) / exp_metric_intra_mean
    
    exp_metric_inter_var /= (rmax_inter_mean/rmin_inter_mean)**2
    exp_metric_inter_var *= 2
    exp_metric_inter_mean = shift  # EX(ln(Rmax/Rmin)) - EX(ln(Rmax/Rmin)) = 0
    exp_metric_inter_cv = np.sqrt(exp_metric_inter_var) / exp_metric_inter_mean
    
    exp_metric_total_var /= (rmax_inter_mean/rmin_inter_mean)**2
    exp_metric_total_var *= 2
    exp_metric_total_mean = shift  # EX(ln(Rmax/Rmin)) - EX(ln(Rmax/Rmin)) = 0
    exp_metric_total_cv = np.sqrt(exp_metric_total_var) / exp_metric_total_mean
    
else:
    raise Exception("Error in metric name !!")

metric_intra_std = metric_denoised_mov.std(axis=1, ddof=1)
metric_intra_mean = metric_denoised_mov.mean(axis=1)
metric_intra_cv = metric_intra_std / metric_intra_mean

metric_inter_std  = metric_intra_mean.std(axis=0, ddof=1)
metric_inter_mean = metric_intra_mean.mean(axis=0)
metric_inter_cv = metric_inter_std / metric_inter_mean

metric_total_std = metric_denoised_mov.reshape(-1, len(wlpivot)).std(axis=0, ddof=1)
metric_total_cv  = metric_total_std / metric_inter_mean

metric_intra_var_mean = np.mean(metric_intra_std**2, axis=0)
metric_inter_var = metric_inter_std**2
metric_total_var = metric_total_std**2

# each intra
# for inter_idx, (intra_mea, intra_exp) in enumerate(zip(metric_intra_cv, exp_metric_intra_cv)):
#     if metricName == "Rmax/Rmin":
#         plt.figure(figsize=figSize)
#         plt.plot(wlpivot, r_intra[inter_idx])
#         plt.xlabel("Wavelength [nm]")
#         plt.ylabel("Correlation coefficient [-]")
#         plt.grid()
#         plt.title(f"Intra-trial - {inter_idx}, Correlation of Rmax and Rmin")
#         plt.show()
    
#     fig, ax = plt.subplots(1, 1, figsize=figSize)
#     ax.plot(wlpivot, intra_mea, linewidth=4, alpha=0.6, label="Measured")
#     ax.plot(wlpivot, intra_exp, linewidth=0.8, label=f"Expected - corr={doCorr}")
#     ax.legend()
#     plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
#     ax.set_xlabel("Wavelength [nm]")
#     ax.set_ylabel("CV [-]")
#     ax.grid()
#     # axtmp = ax.twinx()
#     # axtmp.plot(wlpivot, metric_intra_mean[inter_idx], color=lnm[0].get_color())
#     # axtmp.fill_between(wlpivot, 
#     #                    metric_intra_mean[inter_idx]-metric_intra_std[inter_idx], 
#     #                    metric_intra_mean[inter_idx]+metric_intra_std[inter_idx], 
#     #                    color=lnm[0].get_color(),
#     #                    alpha=0.4)
#     # axtmp.set_ylabel(metricName)
#     ax.set_title(f"Intra-trial variation - {inter_idx}, metric = {metricName}")
#     plt.show()

# mean of intra
# plt.figure(figsize=figSize)
# plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
# plt.plot(wlpivot, metric_intra_cv.mean(axis=0), linewidth=4, alpha=0.6, label="Measured")
# plt.plot(wlpivot, exp_metric_intra_cv.mean(axis=0), linewidth=0.8, label=f"Expected - corr={doCorr}")
# plt.legend()
# plt.xlabel("Wavelength [nm]")
# plt.ylabel("CV [-]")
# plt.grid()
# plt.title(f"Mean of intra-trial variation, metric = {metricName}")
# plt.show()

# inter-trial
# if metricName == "Rmax/Rmin":
#     plt.figure(figsize=(14, 2))
#     plt.plot(wlpivot, r_inter[0])
#     plt.xlabel("Wavelength [nm]")
#     plt.ylabel("Correlation coefficient [-]")
#     plt.grid()
#     plt.title("Inter-trial, Correlation of Rmax and Rmin")
#     plt.show()

# fig, ax = plt.subplots(1, 1, figsize=(14, 2))
# ax.plot(wlpivot, metric_inter_cv, label="Measured")
# ax.plot(wlpivot, exp_metric_inter_cv[0], label=f"Expected - corr={doCorr}")
# plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
# ax.legend()
# ax.set_xlabel("Wavelength [nm]")
# ax.set_ylabel("CV [-]")
# ax.grid()
# ax.set_title(f"Inter-trial variation, metric = {metricName}")
# plt.show()

# compare 6 repeats
plt.figure(figsize=(6, 1.5))
for inter_idx, intra_mean in enumerate(exp_metric_intra_mean):
    plt.plot(wlpivot, intra_mean, label=f"Trial - {inter_idx}")
plt.grid(visible=False)
plt.legend(edgecolor="black", fontsize="small",
           loc='upper right', 
           bbox_to_anchor=(1.22, 1.03))
plt.xlabel("Wavelength [nm]")
plt.ylabel("Rmax/Rmin [-]")
plt.title("Rmax/Rmin (6 repeats)")
plt.show()

metriccorr = np.empty((exp_metric_intra_mean.shape[0], wlcorr.shape[0]))
for idx, metric in enumerate(exp_metric_intra_mean):
    metriccorr[idx] = np.interp(wlcorr, wlpivot, exp_metric_intra_mean[idx])
metriccorrcoef = np.corrcoef(metriccorr.T)[np.triu_indices(metriccorr.T.shape[0], 1)]

rowidx = np.triu_indices(metriccorr.T.shape[0], 1)[0]
colidx = np.triu_indices(metriccorr.T.shape[0], 1)[1]
metriccorrcoef_wl = []
for wlidx in range(len(wlcorr)):
    match = (wlidx == rowidx) | (wlidx == colidx)
    metriccorrcoef_wl.append(np.corrcoef(metriccorr.T)[(rowidx[match], colidx[match])].mean())

plt.figure(figsize=(4, 2.5))
plt.plot(wlcorr, metriccorrcoef_wl, marker=".")
plt.grid(visible=False)
plt.xlabel("Wavelength [nm]")
plt.ylabel("Average correlation coefficient [-]")
plt.title("Rmax/Rmin")
plt.show()


# total
# if metricName == "Rmax/Rmin":
#     plt.figure(figsize=(14, 2))
#     plt.plot(wlpivot, r_total[0])
#     plt.xlabel("Wavelength [nm]")
#     plt.ylabel("Correlation coefficient [-]")
#     plt.grid()
#     plt.title("Total, Correlation of Rmax and Rmin")
#     plt.show()

# plt.figure(figsize=(14, 2))
# plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
# plt.plot(wlpivot, metric_total_cv, label="Measured")
# plt.plot(wlpivot, exp_metric_total_cv[0], label=f"Expected - corr={doCorr}")
# plt.legend()
# plt.xlabel("Wavelength [nm]")
# plt.ylabel("CV [-]")
# plt.grid()
# plt.title(f"Total variation, metric = {metricName}")
# plt.show()

# # compare variance - measured
# plt.figure(figsize=(13, 3))
# plt.plot(wlpivot, metric_total_var, label="Total")
# plt.plot(wlpivot, metric_inter_var, label="Inter")
# plt.plot(wlpivot, metric_intra_var_mean, label="Intra mean")
# plt.legend()
# plt.xlabel("Wavelength [nm]")
# plt.ylabel("Variance [-]")
# plt.grid()
# plt.title(f"Comparison of measured variance, metric = {metricName}")
# plt.show()

# compare variance - measured (estimated inter)
# plt.figure(figsize=(8.4, 2))
# plt.plot(wlpivot, metric_total_var, label="Total")
# plt.plot(wlpivot, metric_total_var - metric_intra_var_mean, label="Inter (through variance subtraction) - probe placement")
# plt.plot(wlpivot, metric_intra_var_mean, label="Intra (mean of 6 repeats)")
# plt.grid(visible=False)
# plt.legend(edgecolor="black", fontsize="small")
# plt.xlabel("Wavelength [nm]")
# plt.ylabel("Variance [-]")
# plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
# # plt.title(f"Comparison of variance, metric = {metricName}, SDS = {nameSet[0].split('_')[1]} mm")
# plt.title(f"SDS = {nameSet[0].split('_')[1]} mm")
# plt.show()

# # compare variance - expected
# plt.figure(figsize=(13, 3))
# plt.plot(wlpivot, exp_metric_total_var[0], label="Total")
# plt.plot(wlpivot, exp_metric_inter_var[0], label="Inter")
# plt.plot(wlpivot, exp_metric_intra_var.mean(axis=0), label="Intra mean")
# plt.legend()
# plt.xlabel("Wavelength [nm]")
# plt.ylabel("Variance [-]")
# plt.grid()
# plt.title(f"Comparison of expected variance, metric = {metricName}")
# plt.show()

# compare variance - expected (estimated inter)
# plt.figure(figsize=(13, 3))
# plt.plot(wlpivot, exp_metric_total_var[0], label="Total")
# plt.plot(wlpivot, exp_metric_total_var[0] - exp_metric_intra_var.mean(axis=0), 
#           label="Inter (estimated)")
# plt.plot(wlpivot, exp_metric_intra_var.mean(axis=0), label="Intra mean")
# plt.legend()
# plt.xlabel("Wavelength [nm]")
# plt.ylabel("Variance [-]")
# plt.grid()
# plt.title(f"Comparison of expected variance, metric = {metricName}")
# plt.show()

# # compare cv - measured
# plt.figure(figsize=(13, 3))
# plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
# plt.plot(wlpivot, metric_total_cv, label="Total")
# plt.plot(wlpivot, metric_inter_cv, label="Inter")
# plt.plot(wlpivot, metric_intra_cv.mean(axis=0), label="Intra mean")
# plt.legend()
# plt.xlabel("Wavelength [nm]")
# plt.ylabel("CV [-]")
# plt.grid()
# plt.title(f"Comparison of Measured CV, metric = {metricName}")
# plt.show()

# # compare cv - expected
# plt.figure(figsize=(13, 3))
# plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
# plt.plot(wlpivot, exp_metric_total_cv[0], label="Total")
# plt.plot(wlpivot, exp_metric_inter_cv[0], label="Inter")
# plt.plot(wlpivot, exp_metric_intra_cv.mean(axis=0), label="Intra mean")
# plt.legend()
# plt.xlabel("Wavelength [nm]")
# plt.ylabel("CV [-]")
# plt.grid()
# plt.title(f"Comparison of Expected CV - corr={doCorr}, metric = {metricName}")
# plt.show()

# compare cv - expected (estimated)
# plt.figure(figsize=(13, 3))
# plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
# plt.plot(wlpivot, exp_metric_total_cv[0], label="Total")
# plt.plot(wlpivot, np.sqrt(exp_metric_total_var[0] - exp_metric_intra_var.mean(axis=0)) / exp_metric_inter_mean, 
#           label="Detector (estimated)")
# plt.plot(wlpivot, np.sqrt(exp_metric_intra_var.mean(axis=0)) / exp_metric_inter_mean, 
#           label="Phy + Sys (mean of 5 repeats)")
# plt.legend(fontsize="large")
# plt.xlabel("Wavelength [nm]", fontsize="large")
# plt.ylabel("CV [-]", fontsize="large")
# plt.grid()
# # plt.title(f"Comparison of Expected CV - corr={doCorr}, metric = {metricName}")
# plt.title(f"Comparison of expected CV, metric = {metricName}", fontsize="x-large")
# plt.show()

# compare cv - measured (estimated)
detectorCV = np.sqrt(metric_total_var - metric_intra_var_mean) / metric_inter_mean
intraCV = np.sqrt(metric_intra_var_mean) / metric_inter_mean
plt.figure(figsize=(8.4, 2))
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
plt.plot(wlpivot, metric_total_cv, 
         label="Total")
plt.plot(wlpivot, detectorCV, 
         label="Inter (through variance subtraction) - probe placement")
plt.plot(wlpivot, intraCV, 
         label="Intra (mean of 6 repeats)")
plt.grid(visible=False)
plt.legend(edgecolor="black", fontsize="small")
plt.xlabel("Wavelength [nm]")
plt.ylabel("CV [-]")
plt.ylim(plt.ylim()[0], 0.016)
# plt.title(f"Comparison of Expected CV - corr={doCorr}, metric = {metricName}")
# plt.title(f"Comparison of CV, metric = {metricName}, SDS = {nameSet[0].split('_')[1]} mm")
plt.title(f"SDS = {nameSet[0].split('_')[1]} mm")
plt.show()

# save
df = pd.DataFrame(np.concatenate((wlpivot[:, None],  metric_total_cv[:, None], 
                                  detectorCV[:, None], intraCV[:, None]), 
                                 axis=1), 
                  columns =["wl [nm]", "total cv [%]", "detector cv [%]", "phy + sys cv [%]"])

df.to_csv(os.path.join(invivo_folder, f"{nameSet[0][:6]}_stability_cv_result.csv"), 
          index=False)

# %% compare stability of different sds
from glob import glob

stabDirSet = glob(os.path.join(invivo_folder, "*stability*result*"))
stabDirSet.sort(key=lambda x: int(x.split("/")[1][4:6]))
stabSet = {}

for stabDir in stabDirSet:
    sds = "_".join(stabDir.split("/")[1].split("_")[:-3])
    stabSet[sds] = pd.read_csv(stabDir)
columns = stabSet[sds].columns
wl = stabSet[sds][columns[0]].values

for ntype in columns[1:]:
    plt.figure(figsize=(8.4, 2))
    for sds, stab in stabSet.items():
        if sds == "sds_20_another":
            plt.plot(wl, stab[ntype], label=f"SDS = {sds[4:6]} mm (another)")
        else:
            plt.plot(wl, stab[ntype], label=f"SDS = {sds[4:6]} mm") 
    if ntype == 'detector cv [%]':
    #     plt.axhline(y = 0.01, color = 'r', linestyle = '--')
        plt.ylim(plt.ylim()[0], plt.ylim()[1]*1.12)
    plt.grid(visible=True)
    plt.legend(edgecolor="black", fontsize="small")
    plt.xlabel("Wavelength [nm]")
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    plt.ylabel("CV [-]")
    plt.grid()
    plt.title(f"Noise - {ntype[:-7].title()}", fontsize="x-large")
    # plt.title("Effect of Probe Placement in different SDS")
    plt.show()






