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
import utils
import os
from glob import glob
import json
import matplotlib as mpl
from matplotlib.lines import Line2D
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science', 'grid'])
# plt.rcParams.update({"mathtext.default": "regular"})
# plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["figure.dpi"] = 600


# %% function

# get estimated instrumental noise
def get_system_noise(S, noise_r, G):
    noise = np.sqrt(noise_r**2 + S/G)
    return noise


# %% parameters
# 20230630_EU_contrast_trend_upward, 20230703_BY_contrast_trend_upward, 20230706_HY_contrast_trend_upward
projectID = "20230426_contrast_investigate_ijvdepth_sdsrange_5to45_g99"
considerAvg = False

inst_noise_charac_file = "shared_files/scs_qepro_system_noise_characterization_formal_equation.json"
caliLine_file = "20230824_phantom_calibration/result/20230731/calibrate_result.csv"

wl_target = np.array([798])  # 732, 798 850 nm

src_fc_set = np.array([1, 2, 4])

sdsObservedRange = [0, 42]

ratioDetCV = 0  # 0.009
ratioPhyCV = 0.01

linestyleSet = ["solid", "dashed", "dashdot"]


# %% load data
# noise
with open(inst_noise_charac_file) as f:
    inst_charac = json.load(f)
inst_popt = list(inst_charac["Coefficient"].values())

# calibration line (sim > measured)
caliLineSet = pd.read_csv(caliLine_file)
wl = np.array(caliLineSet.columns, dtype=int)
wl_target_idx = []
for w in wl_target:
    wl_target_idx.append(np.where(wl == w)[0][0])
caliLineArr = caliLineSet.to_numpy()
caliLine = caliLineArr[:, wl_target_idx]

# average contrast (target to compare)


# sim result
outputPath = f"/media/md703/Expansion/syu/ijv_2_output/{projectID}"
sessionIDSet = glob(os.path.join(outputPath, "*"))
sessionIDSet = [sessionID.split("/")[-1] for sessionID in sessionIDSet]
sessionIDSet.sort(key=lambda x: (-ord(x.split("_")[2][1]), 
                                 float(x.split("_")[-2]), 
                                 -ord(x.split("_")[1][0])))
sessionIDSet = np.array(sessionIDSet).reshape(-1, 2)
sdsPath = glob(os.path.join(outputPath, sessionIDSet[0, 0], "post_analysis", "*"))[0]
with open(sdsPath) as f:
    result = json.load(f)
cv = result["MovingAverageGroupingSampleCV"]
sdsSet = []
for key in cv.keys():
    sdsSet.append(float(key[4:]))
sdsSet = np.array(sdsSet)[sdsObservedRange[0]: sdsObservedRange[1]]
reflectanceSet = {}
for caseType in sessionIDSet:
    case = "_".join(caseType[0].split("_")[2:])
    reflectanceSet[case] = {}
    for idx, ijvType in enumerate(caseType):
        resultPathSet = glob(os.path.join(outputPath, ijvType, "post_analysis", "*mua_skin_50%_fat_50%_muscle_50%_ijv_50%_cca_50%*"))
        ijv = "_".join(ijvType.split("_")[:2])
        reflectanceSet[case][ijv] = {}
        for resultPath in resultPathSet:
            with open(resultPath) as f:
                result = json.load(f)
            mua = "_".join(resultPath.split("_")[-2:])
            mua = mua.split(".")[0]
            reflectanceSet[case][ijv][mua] = np.array(list(result["MovingAverageGroupingSampleMean"].values())[sdsObservedRange[0]: sdsObservedRange[1]])

# load average sim result (target to compare)
if considerAvg:
    rmaxfile = "20230426_contrast_investigate_ijvdepth_sdsrange_5to45_g99/rmax_depth_+0_std_mus_50%_mua_50%.csv"
    rminfile = "20230426_contrast_investigate_ijvdepth_sdsrange_5to45_g99/rmin_depth_+0_std_mus_50%_mua_50%.csv"
    reflectanceSet["average"] = {}
    reflectanceSet["average"]["ijv_col"] = {}
    reflectanceSet["average"]["ijv_dis"] = {}
    reflectanceSet["average"]["ijv_col"]["cca_50%"] = pd.read_csv(rmaxfile)["Reflectance"].values[sdsObservedRange[0]: sdsObservedRange[1]]
    reflectanceSet["average"]["ijv_dis"]["cca_50%"] = pd.read_csv(rminfile)["Reflectance"].values[sdsObservedRange[0]: sdsObservedRange[1]]

# %% process and calculate snr
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
axtmp = ax.twinx()
lns = []
lns += ax.plot(wl, caliLineArr[0, :], label="Coefficient a")
# ax.axvline(x=wl_target[0], color="red", linestyle=linestyleSet[0])
# ax.axvline(x=wl_target[1], color="red", linestyle=linestyleSet[1])
# ax.axvline(x=wl_target[2], color="red", linestyle=linestyleSet[2])
lns += axtmp.plot(wl, caliLineArr[2, :], color="green", label="R square")
# lns += ax.plot(wl_target, caliLineArr[0, wl_target_idx], "o", color="red", 
#                   label=f"{wl_target[0]}, {wl_target[1]}, {wl_target[2]} nm")
ax.grid(visible=False)
axtmp.grid(visible=False)
labels = [l.get_label() for l in lns]
ax.legend(lns, labels, 
           loc='lower right',
           edgecolor="black"
          )
ax.set_xlabel("Wavelength [nm]")
ax.set_ylabel("Coefficient a [-]")
axtmp.set_ylabel("R square [-]")
ax.set_title("$Intensity_{prac}$ = a * $Intensity_{sim}$")
plt.show()

contrastSet = {}
snrSet = {}
for case in reflectanceSet:
    rmax = reflectanceSet[case]["ijv_col"]["cca_50%"]
    rmin = reflectanceSet[case]["ijv_dis"]["cca_50%"]
    ratio = (rmax / rmin)[:, None]
    
    ratioDetNoise = ratio * ratioDetCV
    ratioPhyNoise = ratio * ratioPhyCV
    rmax_mea = rmax[:, None]*caliLine[0, :][None, :] + caliLine[1, :][None, :]
    rmin_mea = rmin[:, None]*caliLine[0, :][None, :] + caliLine[1, :][None, :]
    # plt.plot(sdsSet, rmax_mea)
    # plt.plot(sdsSet, rmin_mea)
    # plt.yscale("log")
    # plt.show()
    rmaxSysNoise = get_system_noise(rmax_mea, *inst_popt)
    rminSysNoise = get_system_noise(rmin_mea, *inst_popt)
    ratioSysNoise = ratio**2 * ((rmaxSysNoise/rmax_mea)**2 + \
                                (rminSysNoise/rmin_mea)**2)
    ratioSysNoise = np.sqrt(ratioSysNoise)
    ratioTotNoise = np.sqrt(ratioDetNoise**2 + 
                            ratioPhyNoise**2 + 
                            ratioSysNoise**2)
    
    contrast = ratio - 1
    snr = contrast / ratioTotNoise
    
    contrastSet[case] = contrast
    snrSet[case] = snr

fig, ax = plt.subplots(2, 2, figsize=(7, 2*7/3))
for idx, case in enumerate(contrastSet.keys()):
    ax[0, 0].plot(sdsSet, contrastSet[case], label=case,
                  color=utils.colorFader(c1="blue", c2="red", mix=idx/(len(contrastSet.keys())-1)))
    for wl_idx, snr in enumerate(snrSet[case].T):
        ax[0, 1].plot(sdsSet, snr, label=f"{wl_target[wl_idx]} nm", linestyle=linestyleSet[wl_idx],
                      color=utils.colorFader(c1="blue", c2="red", mix=idx/(len(contrastSet.keys())-1)))
    ax[1, 0].plot(sdsSet, utils.normalize(contrastSet[case]), label=case,
                  color=utils.colorFader(c1="blue", c2="red", mix=idx/(len(contrastSet.keys())-1)))
    for wl_idx, snr in enumerate(utils.normalize(snrSet[case]).T):
        ax[1, 1].plot(sdsSet, snr, label=f"{wl_target[wl_idx]} nm", linestyle=linestyleSet[wl_idx],
                      color=utils.colorFader(c1="blue", c2="red", mix=idx/(len(contrastSet.keys())-1)))
handles, labels = ax[0, 0].get_legend_handles_labels()
labels = ["5.958 mm ($\mu$-1.6$\sigma$)", "8.13 mm ($\mu$-$\sigma$)", 
          "11.75 mm ($\mu$)", "15.37 mm ($\mu$+$\sigma$)", 
          "18.99 mm ($\mu$+2$\sigma$)"]
fig.legend(handles, labels, edgecolor="black", 
           loc='lower center', bbox_to_anchor=(0.53, -0.08),
           ncol=5, fontsize="x-small",
           title="Distance of IJV upper edge to skin surface")
custom_lines = [Line2D([0], [0], color="black", lw=1, linestyle=linestyleSet[0]),
                Line2D([0], [0], color="black", lw=1, linestyle=linestyleSet[1]),
                Line2D([0], [0], color="black", lw=1, linestyle=linestyleSet[2])]
# ax[0, 1].legend(custom_lines, 
#                 [f"{wl} nm" for wl in wl_target],
#                 edgecolor="black", 
#                 bbox_to_anchor=(1.35, 0.5))
ax[1, 0].set_xlabel("SDS [mm]")
ax[1, 1].set_xlabel("SDS [mm]")
ax[0, 0].set_ylabel("Contrast [-]")
ax[0, 1].set_ylabel("SNR [-]")
ax[1, 0].set_ylabel("Normalized contrast [-]")
ax[1, 1].set_ylabel("Normalized SNR [-]")
ax[0, 0].grid(visible=False)
ax[0, 1].grid(visible=False)
ax[1, 0].grid(visible=False)
ax[1, 1].grid(visible=False)
fig.suptitle(f"Calibration line: {wl_target} nm")
plt.tight_layout()
plt.show()



