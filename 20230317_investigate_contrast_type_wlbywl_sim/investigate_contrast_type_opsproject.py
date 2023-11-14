#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 00:52:11 2022

@author: md703
"""

import numpy as np
import time
import os
import postprocess
import json
from glob import glob
import jdata as jd
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"mathtext.default": "regular"})
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["figure.dpi"] = 300


#%% parameter
outputRoot = "/media/md703/Expansion/syu/ijv_2_output"
infoRoot = "/home/md703/syu/ijv_2"
projectID = "20221129_contrast_investigate_op_sdsrange_3to40"
sessionIDSet = ["ijv_dis_mus_0%", "ijv_col_mus_0%"]  # "ijv_dis_mus_0%", 
photonData = {}
info = jd.load("/media/md703/Expansion/syu/ijv_2_output/20221129_contrast_investigate_op_sdsrange_3to40/ijv_col_mus_0%/mcx_output/ijv_col_mus_0%_0_detp.jdat")["MCXData"]["Info"]

targetWl = 760  # nm
epsilonHbO2HbPath = os.path.join(infoRoot, "shared_files/model_input_related/optical_properties/blood/mua/epsilon_hemoglobin.txt")
conc = 150  # [g/L]
molecularweightHbO2 = 64532  # [g/mol]
molecularweightHb = 64500  # [g/mol]

#%% process blood mua
# read epsilon
epsilonHbO2Hb = pd.read_csv(epsilonHbO2HbPath, sep="\t", names=["wl", "HbO2", "Hb"])
wl = epsilonHbO2Hb["wl"].values
epsilonHbO2 = epsilonHbO2Hb["HbO2"].values
epsilonHb = epsilonHbO2Hb["Hb"].values
# interpolate epsilon to our target wl
epsilonHbO2Used = np.interp(targetWl, wl, epsilonHbO2)  # [cm-1/M]
epsilonHbUsed = np.interp(targetWl, wl, epsilonHb)  # [cm-1/M]
# calculate mua from epsilon
HbO2_mua = 2.303 * epsilonHbO2Used * (conc / molecularweightHbO2) * 0.1  # [1/mm] - from Prahl
Hb_mua = 2.303 * epsilonHbUsed * (conc / molecularweightHb) * 0.1  # [1/mm] - from Prahl


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

# %% investigate contrast type
SijvO2 = np.array([0.65, 0.7, 0.8])
SccaO2 = 0.99
mua_ijv_set = HbO2_mua * SijvO2 + Hb_mua * (1-SijvO2)
mua_cca = HbO2_mua * SccaO2 + Hb_mua * (1-SccaO2)

# %% Rmin
muaPath = os.path.join(infoRoot, projectID, "ijv_dis_mus_0%", "mua_20%.json")
with open(muaPath) as f:
    tmp = json.load(f)
mua = []
for mua_ijv in mua_ijv_set:
    mua.append([tmp["1: Air"],
                tmp["2: PLA"],
                tmp["3: Prism"],
                tmp["4: Skin"],
                tmp["5: Fat"],
                tmp["6: Muscle"],
                mua_ijv,
                mua_ijv,
                mua_cca
                ])
mua = np.array(mua).T

start_time = time.time()
Rmin = postprocess.getSpectrum(mua, photonData["ijv_dis_mus_0%"], info)
t = (time.time() - start_time)
print(f"t = {t}")

# %% Rmax
muaPath = os.path.join(infoRoot, projectID, "ijv_col_mus_0%", "mua_20%.json")
with open(muaPath) as f:
    tmp = json.load(f)
mua = []
for mua_ijv in mua_ijv_set:
    mua.append([tmp["1: Air"],
                tmp["2: PLA"],
                tmp["3: Prism"],
                tmp["4: Skin"],
                tmp["5: Fat"],
                tmp["6: Muscle"],
                tmp["7: Muscle or IJV (Perturbed Region)"],
                mua_ijv,
                mua_cca
                ])
mua = np.array(mua).T

Rmax = postprocess.getSpectrum(mua, photonData["ijv_col_mus_0%"], info)

#%% analyze
rmaxrmin = []
deltalnrmaxrmin = []
for idx, so2 in enumerate(SijvO2):
    rmax = Rmax[:, idx]
    rmin = Rmin[:, idx]
    plt.plot(rmax/rmin)
    plt.ylabel("Rmax / Rmin")
    plt.title(f"SO2: {so2}")
    plt.show()
    
    rmaxrmin.append(rmax/rmin)

contrast = (rmaxrmin[1]-rmaxrmin[0]) / 5
plt.plot(-contrast)
plt.ylabel("delta(Rmax/Rmin) / delta(SO2)")
plt.title("SO2: 0.65 -> 0.7")
plt.show()

contrast = (rmaxrmin[2]-rmaxrmin[1]) / 10
plt.plot(-contrast)
plt.ylabel("delta(Rmax/Rmin) / delta(SO2)")
plt.title("SO2: 0.7 -> 0.8")
plt.show()

for idx, so2 in enumerate(SijvO2):
    rmax = Rmax[:, idx]
    rmin = Rmin[:, idx]
    deltalnrmaxrmin.append(np.log(rmax/rmin))

delta_1 = deltalnrmaxrmin[1] - deltalnrmaxrmin[0]
delta_2 = deltalnrmaxrmin[2] - deltalnrmaxrmin[1]

contrast = delta_1 / 5
plt.plot(-contrast)
plt.ylabel("delta(ln(Rmax/Rmin)) / delta(SO2)")
plt.title("SO2: 0.65 -> 0.7")
plt.show()

contrast = delta_2 / 10
plt.plot(-contrast)
plt.ylabel("delta(ln(Rmax/Rmin)) / delta(SO2)")
plt.title("SO2: 0.7 -> 0.8")
plt.show()

contrast = (delta_2-delta_1) / 5
plt.plot(-contrast)
plt.ylabel("delta_delta(ln(Rmax/Rmin)) / delta_delta(SO2)")
plt.title("delta_SO2: 0.05 -> 0.1")
plt.show()
    