#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 00:52:11 2022

@author: md703
"""

import numpy as np
import os
import postprocess
import utils
import json
from glob import glob
import jdata as jd
import matplotlib.pyplot as plt
plt.rcParams.update({"mathtext.default": "regular"})
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["figure.dpi"] = 300


#%% parameter
outputRoot = "/media/md703/Expansion/syu/ijv_2_output"
infoRoot = "/home/md703/syu/ijv_2"
projectID = "20230317_investigate_contrast_type_wlbywl_sim"
wl = 760
sessionIDSet = [f"ijv_dis_depth_+0_std_op_avg_{wl}nm", 
                f"ijv_col_depth_+0_std_op_avg_{wl}nm"]  # "ijv_dis_mus_0%", "ijv_col_mus_0%"
photonData = {}
info = jd.load(f"/media/md703/Expansion/syu/ijv_2_output/{projectID}/{sessionIDSet[0]}/mcx_output/{sessionIDSet[0]}_0_detp.jdat")["MCXData"]["Info"]


# %% load jdata
for sessionID in sessionIDSet:
    print(f"\n\nsessionID: {sessionID}")
    
    # initialize
    photonData[sessionID] = {}
    photonData[sessionID]["totalSimPhoton"] = 0
    photonData[sessionID]["detid"] = []
    photonData[sessionID]["ppath"] = []
    
    # load
    detOutputPathSet = glob(os.path.join(outputRoot, projectID, sessionID, "mcx_output", "*.jdat"))
    for idx, detOutputPath in enumerate(detOutputPathSet):
        print(f"{idx}, ", end="")
        
        detOutput = jd.load(detOutputPath)["MCXData"]
        photonData[sessionID]["totalSimPhoton"] += detOutput["Info"]["TotalPhoton"]
        photonData[sessionID]["detid"].append(detOutput["PhotonData"]["detid"])
        photonData[sessionID]["ppath"].append(detOutput["PhotonData"]["ppath"])
        
        # check memory, if too tight, stop !
        total_memory, used_memory, free_memory = map(
            int, os.popen('free -t -m').readlines()[-1].split()[1:])
        usedFrac = used_memory / total_memory
        if usedFrac > 0.9:
            raise Exception("Memory nearly explode !!")
        if idx % 50 == 0:            
            print(f"\nRAM memory used: {round(usedFrac*100, 2)}%")

print("\n")

# %% Rmin, Rmax
SijvO2 = np.around(np.arange(0.3, 0.81, 0.1), 2)

SccaO2 = np.array([0.99])

muaRmin = utils.set_mua_matrix(sessionIDSet[0], (SijvO2, SccaO2))
Rmin = postprocess.getSpectrum(muaRmin, photonData[sessionIDSet[0]], info)

muaRmax = utils.set_mua_matrix(sessionIDSet[1], (SijvO2, SccaO2))
Rmax = postprocess.getSpectrum(muaRmax, photonData[sessionIDSet[1]], info)

#%% analyze

# load sds
with open(os.path.join(sessionIDSet[0], "model_parameters.json")) as f:
    modelParam = json.load(f)
sdsSet = [fiber["SDS"] for fiber in modelParam["HardwareParam"]["Detector"]["Fiber"]]
sdsSet.sort()
sdsSet = sdsSet[1:-1]  # delete start and end sds due to moving-average
max_1_sdstholdidx = 24

# Rmax/Rmin & ΔR/R
funcSet = [utils.get_rmax_by_rmin, utils.get_deltaR_by_rmean]
contrastNameSet = ["Rmax / Rmin", "ΔR / R"]
for idx, func in enumerate(funcSet):
    max_1_sds = []
    max_2_sds = []
    lnSet = []
    contrastName = contrastNameSet[idx]
    contrast = func(Rmax, Rmin)
    for idx, so2 in enumerate(SijvO2):
        max_1_sds.append(sdsSet[np.argmax(contrast[:max_1_sdstholdidx, idx])])
        max_2_sds.append(sdsSet[np.argmax(contrast[:, idx])])
        lnSet += plt.plot(sdsSet, contrast[:, idx], 
                          color=utils.colorFader("blue", "red", idx/(SijvO2.shape[0]-1)), 
                          label=f"Sijv$O_2$: {int(so2*100)}%")
        
    # if max sds is the same in different SO2
    max_1_same = np.all(np.array(max_1_sds) == max_1_sds[0])
    max_2_same = np.all(np.array(max_2_sds) == max_2_sds[0])
    if max_1_same and max_2_same:
        title = f"{sessionIDSet[0].split('_')[-1]} - {contrastName}, (Max1, Max2) = ({max_1_sds[0]}, {max_2_sds[0]}) mm"
        labels = [ln.get_label() for ln in lnSet]
    else:
        title = f"{sessionIDSet[0].split('_')[-1]} - {contrastName}, Max is different"
        labels = [ln.get_label() + f" - ({max_1_sds[idx]}, {max_2_sds[idx]}) mm" for idx, ln in enumerate(lnSet)]
    plt.legend(lnSet, labels)
    plt.xlabel("sds [mm]")
    plt.ylabel(f"{contrastName}  [-]")
    plt.title(title)
    plt.show()

# ΔRef / ΔSO2
contrastNameSet = ["ΔRmax / ΔSO2", "ΔRmin / ΔSO2"]
refSet = [Rmax, Rmin]
for idx, ref in enumerate(refSet):
    contrastName = contrastNameSet[idx]
    contrast = utils.get_deltaRef_by_so2(ref, SijvO2)
    contrast = abs(contrast)
    max_1_sds = []
    max_2_sds = []
    lnSet = []
    for idx in range(contrast.shape[1]):
        max_1_sds.append(sdsSet[np.argmax(contrast[:max_1_sdstholdidx, idx])])
        max_2_sds.append(sdsSet[np.argmax(contrast[:, idx])])
        lnSet += plt.plot(sdsSet, contrast[:, idx], 
                          color=utils.colorFader("blue", "red", idx/(SijvO2.shape[0]-1)), 
                          label=f"Sijv$O_2$: {int(SijvO2[idx]*100)}% → {int(SijvO2[idx+1]*100)}%")

    # if max sds is the same in different SO2
    max_1_same = np.all(np.array(max_1_sds) == max_1_sds[0])
    max_2_same = np.all(np.array(max_2_sds) == max_2_sds[0])
    if max_1_same and max_2_same:
        title = f"{sessionIDSet[0].split('_')[-1]} - {contrastName}, (Max1, Max2) = ({max_1_sds[0]}, {max_2_sds[0]}) mm"
        labels = [ln.get_label() for ln in lnSet]
    else:
        title = f"{sessionIDSet[0].split('_')[-1]} - {contrastName}, Max is different"
        labels = [ln.get_label() + f" - ({max_1_sds[idx]}, {max_2_sds[idx]}) mm" for idx, ln in enumerate(lnSet)]
    plt.legend(lnSet, labels)
    plt.xlabel("sds [mm]")
    plt.ylabel(f"{contrastName}  [-]")
    plt.title(title)
    plt.show()

# Δ(contrastType) / ΔSO2
contrastTypeSet = ["rmaxbyrmin", "logOfrmaxbyrmin"]
contrastNameSet = [r"Δ$(\frac{Rmax}{Rmin})$ / Δ$SO_2$", r"Δ$(ln\frac{Rmax}{Rmin})$ / Δ$SO_2$"]
for idx, contrastType in enumerate(contrastTypeSet):
    contrastName = contrastNameSet[idx]
    contrast = utils.get_deltaContrast_by_so2(Rmax, Rmin, contrastType, SijvO2)
    contrast = abs(contrast)
    max_1_sds = []
    max_2_sds = []
    lnSet = []
    for idx in range(contrast.shape[1]):
        max_1_sds.append(sdsSet[np.argmax(contrast[:max_1_sdstholdidx, idx])])
        max_2_sds.append(sdsSet[np.argmax(contrast[:, idx])])
        lnSet += plt.plot(sdsSet, contrast[:, idx], 
                          color=utils.colorFader("blue", "red", idx/(SijvO2.shape[0]-1)),
                          label=f"Sijv$O_2$: {int(SijvO2[idx]*100)}% → {int(SijvO2[idx+1]*100)}%")

    # if max sds is the same in different SO2
    max_1_same = np.all(np.array(max_1_sds) == max_1_sds[0])
    max_2_same = np.all(np.array(max_2_sds) == max_2_sds[0])
    if max_1_same and max_2_same:
        title = f"{sessionIDSet[0].split('_')[-1]} - {contrastName}, (Max1, Max2) = ({max_1_sds[0]}, {max_2_sds[0]}) mm"
        labels = [ln.get_label() for ln in lnSet]
    else:
        title = f"{sessionIDSet[0].split('_')[-1]} - {contrastName}, Max is different"
        labels = [ln.get_label() + f" - ({max_1_sds[idx]}, {max_2_sds[idx]}) mm" for idx, ln in enumerate(lnSet)]
    plt.legend(lnSet, labels)
    plt.xlabel("sds [mm]")
    plt.ylabel(f"{contrastName}  [-]")
    plt.title(title)
    plt.show()