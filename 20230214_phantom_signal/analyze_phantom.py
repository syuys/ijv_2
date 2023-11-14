#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 12:38:30 2023

@author: md703
"""

import numpy as np
import pandas as pd
import os
import json
from glob import glob
from scipy.optimize import curve_fit
import matplotlib as mpl
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science', 'grid'])
plt.rcParams.update({"mathtext.default": "regular"})
# plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["figure.dpi"] = 300
# plt.rcParams["legend.facecolor"] = "white"
# plt.rcParams["legend.edgecolor"] = "black"

# %% parameter
phantomName = "I"
saveNoise = False
doFigLog = False
doReduceNoise = True
reduceType = "MovingAverage"
kernelSize = 23
sdsStartIdxSet = [0]
expType = os.path.join("snr_measure", phantomName)
wlStartIdx = 5  #  520
wlEndIdx = -5   # -355
fileSet = glob(os.path.join(expType, "*"))
c1 = "red"
c2 = "blue"
fileSet.sort(key=lambda x: int("".join(map(str, (map(ord, x.split("_")[2]))))))

def colorFader(c1, c2, mix=0): # fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)

def check_impulse(data, measureType):
    for d in data:
        plt.plot(wl, d)
    plt.xlabel("wl [nm]")
    plt.ylabel("intensity [-]")
    plt.title(f"{measureType}")
    plt.show()


def reduceNoise(wl, kernelSize, Signal, reduceType):
    if reduceType == "Binning":
        shape = Signal[list(Signal.keys())[0]].shape
        groupNum = shape[1]//kernelSize
        # select new wl in low res
        wl = wl[np.arange(groupNum)*kernelSize + (kernelSize-1)//2]
        for measureType in Signal.keys():
            shape = Signal[measureType].shape
            data = np.empty((shape[0], groupNum))
            for idx in range(groupNum):
                data[:, idx] = Signal[measureType][:, kernelSize*idx:kernelSize*(idx+1)].mean(axis=1)
            Signal[measureType] = data
    elif reduceType == "MovingAverage":
        pivot = int((kernelSize-1)/2)
        if pivot >= 1:
            wl = wl[pivot:-pivot]
        for measureType in Signal.keys():
            shape = Signal[measureType].shape
            data = np.empty((shape[0], shape[1]-pivot*2))
            for idx in range(shape[0]):
                data[idx] = np.convolve(Signal[measureType][idx], 
                                        np.ones(kernelSize)/kernelSize, 
                                        mode="valid")
            Signal[measureType] = data
    else:
        raise Exception("Error in reduceType !")
    
    return wl, Signal


# %% analyze signal
Signal = {}

# access wavelength and background
df = pd.read_csv(fileSet[-1], sep="\t", skiprows=14, 
                  usecols = lambda x: "Unnamed" not in x)
print(f"{fileSet[-1]}: \n{df.shape}\n")
wl = df.columns.values.astype(float)[wlStartIdx:wlEndIdx]
Signal["background"] = df.to_numpy()[:, wlStartIdx:wlEndIdx]

# access other signal
sdsSet = []
for idx, file in enumerate(fileSet[:-1]):
    measureType = "_".join(file.split("/")[-1].split("_")[:2])
    sdsSet.append(int(measureType.split("_")[-1]))
    
    df = pd.read_csv(file, sep="\t", skiprows=14, 
                      usecols = lambda x: "Unnamed" not in x)
    print(f"{file}: \n{df.shape}\n")
    
    Signal[measureType] = df.to_numpy()[:, wlStartIdx:wlEndIdx]


# check impulse (salt and pepper noise)
# for measureType, data in Signal.items():
#     check_impulse(data, measureType)

# delete bad signal
Signal["sds_18"] = np.delete(Signal["sds_18"], np.argwhere(Signal["sds_18"].max(axis=1)>2000), axis=0)

# check impulse (salt and pepper noise)
# for measureType, data in Signal.items():
#     check_impulse(data, measureType)


# analyze
if doReduceNoise:
    # wl, Signal = moving_average_lost_res(wl, kernelSize, Signal)
    wl, Signal = reduceNoise(wl, kernelSize, Signal, reduceType=reduceType)

# plot wl resolution
plt.plot(np.diff(wl))
plt.xlabel("Index")
plt.ylabel("Difference [nm]")
plt.title(f"Difference of every two wls. (Number used to do average: {kernelSize})")
plt.show()

# plot background
mean = Signal["background"].mean(axis=0)
std = Signal["background"].std(axis=0, ddof=1)
snr = mean / std

fig, axes = plt.subplots(1, 3, figsize=(15,4))
axes[0].plot(wl, mean, marker=".", linestyle="-")
axes[0].set_xlabel("wl [nm]")
axes[0].set_ylabel("intensity mean [-]")
axes[0].set_title("background, mean")
axes[1].plot(wl, std)
axes[1].set_xlabel("wl [nm]")
axes[1].set_ylabel("intensity std [-]")
axes[1].set_title("background, std")
axes[2].plot(wl, snr)
axes[2].set_xlabel("wl [nm]")
axes[2].set_ylabel("snr [-]")
axes[2].set_title("background, snr = mean / std")
plt.tight_layout()
plt.show()

# plot other signal
for sdsStartIdx in sdsStartIdxSet:
    targetSdsSet = sdsSet[sdsStartIdx:]
    
    # n for gradient plot
    n = len(targetSdsSet)
    for _ in range(2):
        meanSet = []
        stdSet = []
        snrSet = []
        cvSet = []
        fig, axes = plt.subplots(1, 3, figsize=(15,4))
        for idx, sds in enumerate(targetSdsSet):
            measureType = f"sds_{sds}"
            data = Signal[measureType]
            if _ == 1:
                data = data - Signal["background"].mean(axis=0)  # signal - background
            
            # mean
            mean = data.mean(axis=0)
            meanSet.append(mean)
            # std
            std = data.std(axis=0, ddof=1)
            stdSet.append(std)
            # snr & cv
            snr = mean / std
            snrSet.append(snr)
            cv = std / mean
            cvSet.append(cv)
            
            # plot signal mean, std, and snr
            axes[0].plot(wl, mean, linestyle="-", color=colorFader(c1, c2, idx/(n-1)), label=measureType)            
            axes[1].plot(wl, std, color=colorFader(c1, c2, idx/(n-1)), label=measureType)            
            axes[2].plot(wl, snr, color=colorFader(c1, c2, idx/(n-1)), label=measureType)
            
        
        if doFigLog:
            for ax in axes:
                ax.set_yscale("log")
        axes[0].set_xlabel("Wavelength [nm]")
        axes[0].set_ylabel("Mean intensity [counts]")
        axes[0].legend(fontsize=8)
        axes[0].set_title("signal")
        axes[1].set_xlabel("Wavelength [nm]")
        axes[1].set_ylabel("Noise [counts]")
        axes[1].legend(fontsize=8)
        axes[1].set_title("std")
        axes[2].set_xlabel("Wavelength [nm]")
        axes[2].set_ylabel("Signal to Noise ratio [-]")
        axes[2].legend(fontsize=8)
        axes[2].set_title("snr = mean / std")
        if _ == 0:
            fig.suptitle(f"Phantom {phantomName}, sds {targetSdsSet[0]}mm to {targetSdsSet[-1]}mm")
        else:
            fig.suptitle(f"Phantom {phantomName}, sds {targetSdsSet[0]}mm to {targetSdsSet[-1]}mm (SUBTRACT BACKGROUND)")
        plt.tight_layout()
        plt.show()
        
        # plot intensity v.s. (snr & cv)
        if _ == 1:
            fig, axes = plt.subplots(1, 2, figsize=(10,4))
            for idx, sds in enumerate(targetSdsSet):
                axes[0].scatter(meanSet[idx], snrSet[idx], s=8, color=colorFader(c1, c2, idx/(n-1)), label=f"sds_{sds}")
                axes[1].scatter(meanSet[idx], cvSet[idx], s=8, color=colorFader(c1, c2, idx/(n-1)), label=f"sds_{sds}")
            axes[0].set_xlabel("intensity [-]")
            axes[0].set_ylabel("snr [-]")
            axes[1].set_xlabel("intensity [-]")
            axes[1].set_ylabel("cv [-]")
            vals = axes[1].get_yticks()
            axes[1].yaxis.set_major_locator(mticker.FixedLocator(vals))
            axes[1].set_yticklabels(['{:,.1%}'.format(x) for x in vals])
            axes[0].legend()
            axes[1].legend()
            axes[0].set_title("intensity v.s. snr")
            axes[1].set_title("intensity v.s. cv")
            fig.suptitle(f"sds {targetSdsSet[0]}mm to {targetSdsSet[-1]}mm (substract background)")
            plt.tight_layout()
            plt.show()


# curve fit - reflectance v.s. cv
# def get_cv(ref, a, b, c):
#     cv = a*ref**(-b) + c
#     return cv

# meanSet = list(np.array(meanSet).ravel())
# cvSet = list(np.array(cvSet).ravel())
# curve = list(zip(meanSet, cvSet))
# curve.sort(key=lambda x: x[0])
# meanSet, cvSet = zip(*curve)
# meanSet, cvSet = np.array(meanSet), np.array(cvSet)
# popt, pcov = curve_fit(get_cv, meanSet, cvSet)
# print(f"popt: {popt}")
# res = cvSet - get_cv(meanSet, *popt)
# tot = cvSet - np.mean(cvSet)
# ss_res = np.sum(res**2)
# ss_tot = np.sum(tot**2)
# r_sq = 1 - (ss_res / ss_tot)

# plt.plot(meanSet, cvSet, ".", label="Experiment")
# plt.plot(meanSet, get_cv(meanSet, *popt), label=f"Fit, $R^2$={np.around(r_sq, 2)}")
# plt.legend()
# plt.xlabel("Mean Intensity [counts]")
# plt.ylabel("CV [-]")
# plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
# plt.title("Mean Intensity v.s. CV")
# plt.show()

# plt.plot(meanSet[:2250], cvSet[:2250], ".", label="Experiment")
# plt.plot(meanSet[:2250], get_cv(meanSet, *popt)[:2250], label="Fit")
# plt.grid()
# plt.legend()
# plt.xlabel("Mean Intensity [counts]")
# plt.ylabel("CV [-]")
# plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
# plt.title("Mean Intensity v.s. CV  (Local)")
# plt.show()


# %% new
# noise raw
fig, ax = plt.subplots(1, 2, figsize=(6.5, 2.8))
for idx, (mean, std) in enumerate(zip(meanSet, stdSet)):
    ax[0].plot(wl, mean, color=colorFader(c1, c2, idx/(len(meanSet)-1)), 
                  linestyle="-", label=f"SDS={targetSdsSet[idx]}mm",
                  )
    ax[1].plot(wl, std, color=colorFader(c1, c2, idx/(len(meanSet)-1)), 
                  linestyle="-", label=f"SDS={targetSdsSet[idx]}mm",
                  )
ax[0].set_xlabel("Wavelength [nm]")
ax[0].set_ylabel("Mean intensity [counts]")
ax[0].grid(visible=False)
ax[0].text(0.04, 0.87, "(a)", fontsize="x-large", horizontalalignment='left',
          verticalalignment='bottom', transform=ax[0].transAxes)
ax[1].set_xlabel("Wavelength [nm]")
ax[1].set_ylabel("Noise [counts]")
ax[1].grid(visible=False)
ax[1].text(0.04, 0.87, "(b)", fontsize="x-large", horizontalalignment='left',
          verticalalignment='bottom', transform=ax[1].transAxes)
handles, labels = ax[0].get_legend_handles_labels()
fig.legend(handles, labels, facecolor="white", edgecolor="black", 
           # framealpha=0.8,
           loc='lower center', bbox_to_anchor=(0.5, -0.05),
           ncol=6, fontsize=6.7)
plt.tight_layout()
plt.show()

# noise process and curve fit - reflectance v.s. noise
for idx, (mean, std) in enumerate(zip(meanSet, stdSet)):
    plt.scatter(meanSet[idx], stdSet[idx], s=8, 
                color=colorFader(c1, c2, idx/(len(meanSet)-1)), 
                label=f"SDS={targetSdsSet[idx]}mm")

def get_noise(S, noise_r, G):
    noise = np.sqrt(noise_r**2 + S/G)
    return noise

meanSet = list(np.array(meanSet).ravel())
stdSet = list(np.array(stdSet).ravel())
curve = list(zip(meanSet, stdSet))
curve.sort(key=lambda x: x[0])
meanSet, stdSet = zip(*curve)
meanSet, stdSet = np.array(meanSet), np.array(stdSet)
popt, pcov = curve_fit(get_noise, meanSet, stdSet)
print(f"popt: {popt}")
res = stdSet - get_noise(meanSet, *popt)
tot = stdSet - np.mean(stdSet)
ss_res = np.sum(res**2)
ss_tot = np.sum(tot**2)
r_sq = 1 - (ss_res / ss_tot)

plt.plot(meanSet, get_noise(meanSet, *popt), color="black", linewidth=3,
         label="N = (N$^2_R$ + S/G)$^{1/2}$" + f",  $R^2$={np.around(r_sq, 2)}")
# plt.plot(meanSet, stdSet, ".", label="Experiment")
# plt.plot(meanSet, get_noise(meanSet, *popt), label=F"$R^2$={np.around(r_sq, 2)}")
leg = plt.legend(facecolor="white", edgecolor="black", 
           # framealpha=0.8,
            # loc='upper right', 
            bbox_to_anchor=(1.01, 1.03),
           # ncol=6, 
            fontsize="x-small"
           )
plt.xlabel("Mean Intensity [counts]")
plt.ylabel("Noise [counts]")
# plt.title("Mean Intensity v.s. Noise")
plt.grid(visible=False)
plt.show()

# if saveNoise:
#     noise = {
#             "__comment__": "The  coefficient is based on: CV = a*Reflectance^(-b) + c. Reflectance is the intensity measured by QEPro. CV is the expected noise w.r.t the reflectance.", 
#             "source": os.getcwd().split("/")[-1],
#             "Integration time [s]": 0.1,
#             f"Pixel number of {reduceType}": kernelSize,
#             "R square": r_sq,
#             "Coefficient": {
#                 "a": popt[0],
#                 "b": popt[1],
#                 "c": popt[2]
#                 }
#         }
#     with open("/home/md703/syu/ijv_2/shared_files/system_noise_characterization.json", "w") as f:
#         json.dump(noise, f, indent=4)

if saveNoise:
    noise = {
            "__comment__": "The  coefficient is based on: noise = np.sqrt(noise_r**2 + S/G). S is the intensity measured by QEPro. Noise is the expected noise w.r.t the reflectance.", 
            "source": os.getcwd().split("/")[-1],
            "Integration time [s]": 0.1,
            f"Pixel number of {reduceType}": kernelSize,
            "R square": r_sq,
            "Coefficient": {
                "noise_r": popt[0],
                "G":       popt[1]
                }
        }
    with open("/home/md703/syu/ijv_2/shared_files/system_noise_characterization_formal_equation.json", "w") as f:
        json.dump(noise, f, indent=4)

