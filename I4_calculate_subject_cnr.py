#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  2 16:40:27 2023

@author: md703
"""

import numpy as np
import pandas as pd
from scipy.signal import convolve
from scipy.interpolate import interp1d
import itertools
import utils
import os
from glob import glob
import json
import matplotlib as mpl
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science', 'grid'])
# plt.rcParams.update({"mathtext.default": "regular"})
# plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["figure.dpi"] = 600


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
def get_system_noise(S, noise_r, G):
    noise = np.sqrt(noise_r**2 + S/G)
    return noise


def binning(data, num):
    gr = data.shape[1] // num
    newdata = np.empty((data.shape[0], gr, data.shape[2]))
    for idx in range(gr):
        newdata[:, idx, :] = data[:, num*idx:num*(idx+1), :].mean(axis=1)
    return newdata


def flip(items, ncol):
    return itertools.chain(*[items[i::ncol] for i in range(ncol)])


# %% parameters
# 20230630_EU_contrast_trend_upward, 20230703_BY_contrast_trend_upward, 20230706_HY_contrast_trend_upward
invivo_folder = "20230706_HY_contrast_trend_upward"
rmaxfileSet = glob(os.path.join(invivo_folder, "*rmax.csv"))
rminfileSet = glob(os.path.join(invivo_folder, "*rmin.csv"))
subject = invivo_folder.split("_")[1]
wl = np.array(pd.read_csv(rmaxfileSet[0]).columns, dtype=float)
pixel_res = np.diff(wl).mean()
sdsSet = []

figSize = (12, 4)
inst_noise_charac_file = "shared_files/system_noise_characterization_formal_equation.json"
wl_test = [730, 760, 780, 810, 850]
wl_slide_num_base = 23
wl_slide_num_set = [wl_slide_num_base, 
                      # wl_slide_num_base*2+1
                      ]
peak_slide_num_set = np.array([1, 5, 10, 15])  # for observing Rmax/Rmin variation within one experiment
peak_t_num = 60  # initial (600*0.1)
src_fc_set = np.array([1, 2, 4])


# %% load, sds, denoised rmax and rmin, noise profile
rmaxfileSet.sort(key=lambda x: float(x.split("/")[-1].split("_")[1]))  # by sds
rminfileSet.sort(key=lambda x: float(x.split("/")[-1].split("_")[1]))  # by sds

# rmax and rmin (dict -> ndarray)
rmaxSDSdict = {}
rminSDSdict = {}
for rmaxfile, rminfile in zip(rmaxfileSet, rminfileSet):
    # sds
    sds = rmaxfile.split("/")[-1].split("_")[1]
    sdsSet.append(float(sds))
    
    rmaxSDSdict[sds] = pd.read_csv(rmaxfile).to_numpy()
    rminSDSdict[sds] = pd.read_csv(rminfile).to_numpy()
    
    # update peak_t_num
    t_num = min(rmaxSDSdict[sds].shape[0],
                rminSDSdict[sds].shape[0])
    if t_num < peak_t_num:
        peak_t_num = t_num

sdsSet = np.array(sdsSet).astype(int)
rmaxSDS = np.empty((len(rmaxSDSdict.keys()), peak_t_num, len(wl)))
rminSDS = np.empty((len(rmaxSDSdict.keys()), peak_t_num, len(wl)))
for idx, (rmax, rmin) in enumerate(zip(rmaxSDSdict.values(), rminSDSdict.values())):
    rmaxSDS[idx] = rmax[:peak_t_num]
    rminSDS[idx] = rmin[:peak_t_num]

# noise
with open(inst_noise_charac_file) as f:
    inst_charac = json.load(f)
inst_popt = list(inst_charac["Coefficient"].values())
             


# %% calculate contrast thru rmax and rmin
# contrastMean = rmaxSDS.mean(axis=(1, 2)) / rminSDS.mean(axis=(1, 2))
for wl_slide_num in wl_slide_num_set:
    # slide wl
    wl_slide = wl[(wl_slide_num-1)//2:-(wl_slide_num-1)//2]
    rmaxSDSslide = convolve(rmaxSDS, 
                            np.ones((1, 1, wl_slide_num))/wl_slide_num, 
                            mode='valid')
    rminSDSslide = convolve(rminSDS, 
                            np.ones((1, 1, wl_slide_num))/wl_slide_num, 
                            mode='valid')
    print(f"rmaxSDSslide.shape: {rmaxSDSslide.shape}")
    print(f"rminSDSslide.shape: {rminSDSslide.shape}")
    
    
    # examine noise of each wl, each sds
    # rmaxSDSslide_ex_mean = rmaxSDSslide.mean(axis=1)
    # rminSDSslide_ex_mean = rminSDSslide.mean(axis=1)
    # ratio_ex_EXmean = rmaxSDSslide_ex_mean / rminSDSslide_ex_mean
    # rmaxSDSslide_ex_EXsys_noise = get_system_noise(rmaxSDSslide_ex_mean, 
    #                                                 *inst_popt)
    # rminSDSslide_ex_EXsys_noise = get_system_noise(rminSDSslide_ex_mean, 
    #                                                 *inst_popt)
    # rmaxSDSslide_ex_EXsys_cv = rmaxSDSslide_ex_EXsys_noise / rmaxSDSslide_ex_mean
    # rminSDSslide_ex_EXsys_cv = rminSDSslide_ex_EXsys_noise / rminSDSslide_ex_mean
    # ratio_ex_EXsys_noise = ratio_ex_EXmean**2 * \
    #                                 (rmaxSDSslide_ex_EXsys_cv**2 + \
    #                                   rminSDSslide_ex_EXsys_cv**2)
    # ratio_ex_EXsys_noise = np.sqrt(ratio_ex_EXsys_noise)
    # ratio_ex_EXsys_cv = ratio_ex_EXsys_noise / ratio_ex_EXmean
    
    # ratio_ex = rmaxSDSslide / rminSDSslide
    # ratio_ex_total_noise = ratio_ex.std(axis=1, ddof=1)
    # ratio_ex_mean = ratio_ex.mean(axis=1)
    # ratio_ex_total_cv = ratio_ex_total_noise / ratio_ex_mean
    
    # ratio_ex_physio_noise = np.sqrt(ratio_ex_total_noise**2 - ratio_ex_EXsys_noise**2)
    # ratio_ex_physio_cv = ratio_ex_physio_noise / ratio_ex_mean
    # for sds_idx, sds_cv in enumerate(ratio_ex_total_cv):
    #     plt.plot(wl_slide, sds_cv, label="Measured total noise")
    #     plt.plot(wl_slide, ratio_ex_physio_cv[sds_idx], label="Physio noise")
    #     plt.plot(wl_slide, ratio_ex_EXsys_cv[sds_idx], label="Expected sys noise")
    #     plt.legend()
    #     plt.xlabel("wavelength [nm]")
    #     plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    #     plt.ylabel("cv [-]")
    #     plt.title(f"{sdsSet[sds_idx]} mm")
    #     plt.show()
    
    # sds_obs_num = 6
    # plt.figure(figsize=(8, 1.2))
    # for sds_idx, phy in enumerate(ratio_ex_physio_cv[:sds_obs_num]):
    #     plt.plot(wl_slide, phy, label=f"SDS={sdsSet[sds_idx]}mm",
    #               color=utils.colorFader("red", "blue", sds_idx/(sds_obs_num-1)))
    # handles, labels = plt.gca().get_legend_handles_labels()
    # plt.legend(flip(handles, 3), flip(labels, 3) ,edgecolor="black", loc='lower center', 
    #            bbox_to_anchor=(0.5, -1.1),
    #            ncol=3, fontsize="small")
    # # plt.legend(edgecolor="black", 
    # #             bbox_to_anchor=(1.01, 1.06),
    # #             fontsize="small")
    # plt.xlabel("Wavelength [nm]")
    # plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    # plt.ylabel("CV [-]")
    # title = f"{subject} - Comparision of physio noise in different sds"
    # plt.title(subject)
    # plt.grid(visible=False)
    # plt.show()
    # print(title)
    
    # evaluate practical noise
    time = [1, 5, 10, 15]
    ratioIntraNoise_prac = np.empty((len(time), len(sdsSet), len(wl_slide)))
    for idx, t in enumerate(time):
        ratioIntraNoise_prac[idx] = np.std(binning(rmaxSDSslide/rminSDSslide, t), 
                                        ddof=1, axis=1)
    ratioIntraNoise_prac_wl = np.empty((len(time), len(sdsSet), len(wl_test)))
    for idx, data in enumerate(ratioIntraNoise_prac):
        f_noise = interp1d(wl_slide, data, axis=1)
        ratioIntraNoise_prac_wl[idx] = f_noise(wl_test)
    
    # average time
    rmaxSDSslide = rmaxSDSslide.mean(axis=1)
    rminSDSslide = rminSDSslide.mean(axis=1)    
    
    # interpolate and calculate contrast
    f_max = interp1d(wl_slide, rmaxSDSslide, axis=1)
    f_min = interp1d(wl_slide, rminSDSslide, axis=1)
    # broadcast to src-enhance dim
    rmaxSDSslideWl = f_max(wl_test)*src_fc_set[:, None, None]
    rminSDSslideWl = f_min(wl_test)*src_fc_set[:, None, None]
    # reduce dim of ratio
    ratioWl = (rmaxSDSslideWl / rminSDSslideWl)[0]
    contrastWl = ratioWl - 1
    
    # evaluate total noise and cnr
    ratioDetCV = np.array([0, 0.005, 0.01, 0.015])  # 0.015
    ratioDetNoise = ratioWl * ratioDetCV[:, None, None]
    ratioPhyCV = 0.01
    ratioPhyNoise = ratioWl * ratioPhyCV
    rmaxSysNoise = get_system_noise(rmaxSDSslideWl, *inst_popt)
    rminSysNoise = get_system_noise(rminSDSslideWl, *inst_popt)
    # rmaxSysNoise /= np.sqrt(peak_slide_num_set)[:, None, None]
    # rminSysNoise /= np.sqrt(peak_slide_num_set)[:, None, None]
    ratioSysNoise = ratioWl**2 * ((rmaxSysNoise/rmaxSDSslideWl)**2 + \
                                  (rminSysNoise/rminSDSslideWl)**2)
    ratioSysNoise = np.sqrt(ratioSysNoise)
    # broadcast to heart-beat dim
    ratioIntraNoise = np.sqrt(ratioPhyNoise**2 + 
                              ratioSysNoise**2) / \
                        np.sqrt(peak_slide_num_set)[:, None, None, None]  # broadcast to different time window
    ratioTotNoise = np.sqrt(ratioDetNoise.reshape(4, 1, 1, 14, 5)**2 + 
                            ratioIntraNoise**2)
    
    # practical total noise and cnr
    # ratioTotNoise_prac = np.sqrt(ratioDetNoise**2 + 
    #                         ratioIntraNoise_prac_wl**2)
    # cnrWl_prac = contrastWl / ratioTotNoise_prac
    
    # [DetCV, HeartBeats, srcEnhanceFcs, SDSnum, WLnum]
    cnrWlTime = contrastWl / ratioTotNoise  # ratio noise is equivalent to contrast noise
    
    # plot contrast 
    plt.figure(figsize=(4.5, 2.5))
    for wlidx, contrast in enumerate(contrastWl.T):
        plt.plot(sdsSet, contrast, label=f"{wl_test[wlidx]} nm",
                 color=utils.colorFader("blue", "red", wlidx/(len(wl_test)-1)))
    plt.legend(edgecolor="black", fontsize="small")
    plt.xticks(sdsSet)
    plt.xlabel("SDS [mm]")
    plt.ylabel("Experimental contrast [-]")
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    plt.grid()
    # plt.title(f"{subject} - Signal v.s. SDS, wl res = {round(wl_slide_num*pixel_res, 1)} nm")
    plt.title(f"Subject: {subject}")
    plt.show()
    
    # plot cnr - for DetCV=0, heartbeat num = 1, srcfc = 1
    plt.figure(figsize=(4.2, 2.7))
    for wlidx, cnr in enumerate(cnrWlTime[0][0][0].T):
    # for wlidx, cnr in enumerate(cnrWlTime.T):
        plt.plot(sdsSet, cnr, label=f"{wl_test[wlidx]} nm",
                 linestyle="--",
                 color=utils.colorFader("blue", "red", wlidx/(len(wl_test)-1)))
    plt.plot(sdsSet, cnrWlTime[0][0][0].mean(axis=1), 
             linewidth=2, color="black",
             label="Average"
             )
    plt.text(0.75, 0.7, "Flat IJV\narea", 
              color="navy",
              fontsize="large", horizontalalignment='left',
              verticalalignment='bottom', transform=plt.gca().transAxes,
              )
    plt.ylim(plt.ylim()[0], plt.ylim()[1])
    plt.xlim(plt.xlim()[0], plt.xlim()[1])
    plt.fill_between([18, plt.xlim()[1]], y1=plt.ylim()[0], y2=plt.ylim()[1], alpha=0.2)
    plt.legend(edgecolor="black", fontsize="small",
               # loc='upper right', 
               bbox_to_anchor=(1.3, 1.03))
    plt.grid(visible=False)
    plt.xticks(sdsSet)
    plt.xlabel("SDS [mm]")
    plt.ylabel("SNR [-]")
    # plt.title(f"{subject} - SNR v.s. SDS, wl res = {round(wl_slide_num*pixel_res, 1)} nm")
    plt.title(f"Subject: {subject}")
    plt.show()
    
    # plot different detector cv
    # plt.figure(figsize=(4.2, 2.7))
    # for detidx, cnr in enumerate(cnrWlTime[:, 0, 0, :, :]):
    #     plt.plot(sdsSet, cnr.mean(axis=1), label=f"Probe CV = {ratioDetCV[detidx]*100}\%",
    #              linestyle="-",
    #              color=utils.colorFader("blue", "red", detidx/(len(ratioDetCV)-1)))
    # plt.ylim(plt.ylim()[0], plt.ylim()[1])
    # plt.xlim(plt.xlim()[0], plt.xlim()[1])
    # plt.fill_between([18, plt.xlim()[1]], y1=plt.ylim()[0], y2=plt.ylim()[1], alpha=0.2)
    # plt.text(0.75, 0.7, "Flat IJV\narea", 
    #          color="navy",
    #          fontsize="large", horizontalalignment='left',
    #          verticalalignment='bottom', transform=plt.gca().transAxes,
    #          )
    # plt.legend(edgecolor="black", fontsize="small",
    #             loc='upper right', 
    #            bbox_to_anchor=(1.47, 1.03))
    # plt.grid(visible=False)
    # plt.xticks(sdsSet)
    # plt.xlabel("SDS [mm]")
    # plt.ylabel("SNR [-]")
    # # plt.title(f"{subject} - SNR v.s. SDS, wl res = {round(wl_slide_num*pixel_res, 1)} nm")
    # plt.title(f"Subject: {subject}")
    # plt.show()
    
    # noise model validation (compare standard deviation between model & prac)
    # plt.plot(sdsSet, ratioTotNoise[0][0][0].mean(axis=1), 
    #           label="Model")
    # plt.plot(sdsSet, ratioIntraNoise_prac_wl[0].mean(axis=1), 
    #           label="Measurement")    
    # plt.legend(edgecolor="black", fontsize="small")
    # plt.xticks(sdsSet)
    # plt.xlabel("SDS [mm]")
    # plt.ylabel("$\sigma_{Rmax/Rmin}$ [-]")
    # plt.grid()
    # # plt.title(f"{subject} - noise model validation")
    # plt.title(f"Subject: {subject}")
    # plt.show()
    
    # short sds noise model validation
    # plt.figure(figsize=(3.5, 3.5/4))
    # end = 7
    # plt.plot(sdsSet[:end], ratioTotNoise[0][0][0].mean(axis=1)[:end])
    # plt.plot(sdsSet[:end], ratioIntraNoise_prac_wl[0].mean(axis=1)[:end])    
    # plt.grid(visible=False)
    # plt.xticks([10, 22])
    # plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    # # plt.xlabel("SDS [mm]")
    # # plt.ylabel("Variance [-]")
    # # plt.title("Noise decomposition - value")
    # plt.show()
    
    # noise model validation (calculate the ratio)
    # plt.plot(sdsSet, ratioTotNoise[0][0][0].mean(axis=1) / ratioIntraNoise_prac_wl[0].mean(axis=1))
    # plt.xticks(sdsSet)
    # plt.xlabel("SDS [mm]")
    # plt.ylabel("$\sigma_{model}$ / $\sigma_{measurement}$ [-]")
    # plt.grid()
    # plt.title(f"Subject: {subject}")
    # plt.show()
    
    # plot cnr_prac - for heartbeat num = 1
    # for wlidx, cnr in enumerate(cnrWl_prac[0].T):
    #     plt.plot(sdsSet, cnr, label=f"{wl_test[wlidx]} nm",
    #              color=utils.colorFader("blue", "red", wlidx/(len(wl_test)-1)))
    # plt.legend(fontsize="small")
    # plt.xticks(sdsSet)
    # plt.xlabel("SDS [mm]")
    # plt.ylabel("SNR (practical) [-]")
    # plt.grid()
    # plt.title(f"{subject} - SNR (practical) v.s. SDS, wl res = {round(wl_slide_num*pixel_res, 1)} nm")
    # plt.show()
    
    # compare cnr_prac and cnr_pred
    # plt.figure(figsize=(4, 2.5))
    # plt.plot(sdsSet, cnrWl_prac[0].mean(axis=1), label="Practice")
    # plt.plot(sdsSet, cnrWlTime[0][0].mean(axis=1), label="Model")
    # plt.legend(fontsize="small")
    # plt.xticks(sdsSet)
    # plt.xlabel("SDS [mm]")
    # plt.ylabel("SNR [-]")
    # plt.grid()
    # plt.title(f"{subject} - SNR v.s. SDS, wl res = {round(wl_slide_num*pixel_res, 1)} nm")
    # plt.show()
    
    # compare contrast, noise
    # plt.plot(sdsSet, contrastWl.mean(axis=1), label="contrast")
    # # plt.plot(sdsSet, ratioWl.mean(axis=1), label="contrast")
    # plt.plot(sdsSet, ratioTotNoise.mean(axis=1), label="noise")
    # plt.legend()
    # plt.xticks(sdsSet)
    # plt.xlabel("SDS [mm]")
    # plt.ylabel("[-]")
    # plt.grid()
    # plt.title("Comparison of contrast and noise")
    # plt.show()
    
    # plot cnr - compare different peak
    plt.figure(figsize=(4.4, 2.8))
    for timeidx, cnrWl in enumerate(cnrWlTime[0, :, 0, :, :]):
        plt.plot(sdsSet, cnrWl.mean(axis=1), 
                  label=f"Num of heart beat = {peak_slide_num_set[timeidx]}",
                  color=utils.colorFader("black", "red", timeidx/(cnrWlTime.shape[0]-1)))
    plt.text(0.75, 0.7, "Flat IJV\narea", 
              color="navy",
              fontsize="large", horizontalalignment='left',
              verticalalignment='bottom', transform=plt.gca().transAxes,
              )
    plt.legend(edgecolor="black", fontsize="small", 
               bbox_to_anchor=(1.55, 1.03))
    plt.grid(visible=False)
    plt.ylim(plt.ylim()[0], plt.ylim()[1])
    plt.xlim(plt.xlim()[0], plt.xlim()[1])
    plt.fill_between([18, plt.xlim()[1]], y1=plt.ylim()[0], y2=plt.ylim()[1], alpha=0.2)
    plt.xticks(sdsSet)
    plt.xlabel("SDS [mm]")
    plt.ylabel("SNR [-]")
    # plt.title(f"{subject} - SNR v.s. SDS, wl res = {round(wl_slide_num*pixel_res, 1)} nm")
    plt.title(f"Subject: {subject}")
    plt.show()
    
    # plot cnr - compare different peak [practice v.s model]
    # plt.figure(figsize=(4.4, 2.8))
    # for timeidx, cnrWl in enumerate(cnrWlTime[:, 0, :, :]):
    #     plt.plot(sdsSet, cnrWl.mean(axis=1), 
    #              label=f"num of heart beat = {peak_slide_num_set[timeidx]}",
    #              color=utils.colorFader("black", "red", timeidx/(cnrWlTime.shape[0]-1)))
    # plt.legend(fontsize="small", bbox_to_anchor=(1.03, 1.02))
    # plt.xticks(sdsSet)
    # plt.xlabel("SDS [mm]")
    # plt.ylabel("SNR [-]")
    # plt.grid()
    # plt.title(f"{subject} - Model")
    # # plt.title(f"{subject} - SNR v.s. SDS, wl res = {round(wl_slide_num*pixel_res, 1)} nm")
    # plt.show()
    
    # plt.figure(figsize=(4.4, 2.8))
    # for timeidx, cnrWl in enumerate(cnrWl_prac):
    #     plt.plot(sdsSet, cnrWl.mean(axis=1), 
    #              label=f"num of heart beat = {time[timeidx]} (practice)",
    #              color=utils.colorFader("black", "red", timeidx/(cnrWl_prac.shape[0]-1)),
    #              linestyle="-")
    # plt.legend(fontsize="small", bbox_to_anchor=(1.03, 1.02))
    # plt.xticks(sdsSet)
    # plt.xlabel("SDS [mm]")
    # plt.ylabel("SNR [-]")
    # plt.grid()
    # plt.title(f"{subject} - Practice")
    # # plt.title(f"{subject} - SNR v.s. SDS, wl res = {round(wl_slide_num*pixel_res, 1)} nm")
    # plt.show()
    
    # plot cnr - compare different srcfc
    # for srcidx, cnrWl in enumerate(cnrWlTime[0, :, :, :]):
    #     plt.plot(sdsSet, cnrWl.mean(axis=1), 
    #              label=f"src fc = {src_fc_set[srcidx]}",
    #              color=utils.colorFader("black", "red", srcidx/(cnrWlTime.shape[1]-1)))
    # plt.legend(fontsize="small")
    # plt.xticks(sdsSet)
    # plt.xlabel("SDS [mm]")
    # plt.ylabel("SNR [-]")
    # plt.grid()
    # plt.title(f"{subject} - SNR v.s. SDS, wl res = {round(wl_slide_num*pixel_res, 1)} nm")
    # plt.show()
    
    # decompose noise
    # plt.figure(figsize=(3.3, 2.2))
    # # plt.plot(sdsSet, (ratioDetNoise**2).mean(axis=1), label="Probe")   
    # plt.plot(sdsSet, (ratioPhyNoise**2).mean(axis=1), label="Phy")
    # plt.plot(sdsSet, (ratioSysNoise[0]**2).mean(axis=1), label="Sys")   
    # plt.legend(edgecolor="black", fontsize="small")
    # plt.grid(visible=False)
    # plt.xticks(sdsSet)
    # plt.xlabel("SDS [mm]")
    # plt.ylabel("Variance [-]")
    # # plt.title("Noise decomposition - value")
    # plt.title(f"Subject: {subject}")
    # plt.show()
    
    # decompose noise (short sds)
    # plt.figure(figsize=(2.6, 2.6/4))
    # end = 9
    # # plt.plot(sdsSet[:end], (ratioDetNoise**2).mean(axis=1)[:end], label="Probe")   
    # plt.plot(sdsSet[:end], (ratioPhyNoise**2).mean(axis=1)[:end], label="Phy")
    # plt.plot(sdsSet[:end], (ratioSysNoise[0]**2).mean(axis=1)[:end], label="Sys")   
    # # plt.legend(edgecolor="black", fontsize="small")
    # plt.grid(visible=False)
    # plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    # plt.xticks([10, sdsSet[end-1]])
    # # plt.xlabel("SDS [mm]")
    # # plt.ylabel("Variance [-]")
    # # plt.title("Noise decomposition - value")
    # plt.show()
    
    # relative variance
    # plt.figure(figsize=(3.3, 2.2))
    # # plt.plot(sdsSet, (ratioDetNoise**2/ratioTotNoise[0][0]**2).mean(axis=1), label="Probe")   
    # plt.plot(sdsSet, (ratioPhyNoise**2/ratioTotNoise[0][0][0]**2).mean(axis=1), label="Phy")
    # plt.plot(sdsSet, (ratioSysNoise[0]**2/ratioTotNoise[0][0][0]**2).mean(axis=1), label="Sys")   
    # plt.legend(edgecolor="black", fontsize="small")
    # plt.grid(visible=False)
    # plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    # plt.xticks(sdsSet)
    # plt.xlabel("SDS [mm]")
    # plt.ylabel("Relative variance [-]")
    # # plt.title("Noise decomposition - relative percentage")
    # plt.title(f"Subject: {subject}")
    # plt.show()
    
    # plt.plot(sdsSet, ratioIntraNoise[0][0].mean(axis=1), label="intra")
    # plt.plot(sdsSet, ratioTotNoise[0][0].mean(axis=1), label="intra + detector")
    # plt.legend(fontsize="small")
    # plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    # plt.xticks(sdsSet)
    # plt.xlabel("SDS [mm]")
    # plt.ylabel("Noise level [-]")
    # plt.grid()
    # plt.title("Level of noise")
    # plt.show()

