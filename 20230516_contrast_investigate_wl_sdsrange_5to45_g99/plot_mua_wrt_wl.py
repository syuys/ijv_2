#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 15:14:51 2023

@author: md703
"""

import numpy as np
import pandas as pd
import json
from glob import glob
import os
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science', 'grid'])
plt.rcParams["figure.dpi"] = 600

# %% parameters
coefFile = "/home/md703/syu/ijv_2/shared_files/model_input_related/optical_properties/coefficients_in_cm-1_OxyDeoxyConc150.csv"
wlSet = np.linspace(725, 875, 151)
wlTarSet = [730, 760, 780, 810, 850]
tissueComTemPath = "tissue_composition_template_ijv_0.7.json"
tissueSet = [
            # "skin", "fat", "muscle", 
            "ijv", "cca"
             ]
tissueMua = np.empty((len(tissueSet), len(wlSet)))
tissueMcx = ["4: Skin", "5: Fat", "6: Muscle",
             "8: IJV", "9: CCA"]

def calculate_mua(b, s, w, f, m, c, oxy, deoxy, water, fat, melanin, collagen):
    if b+w+f+m+c == 1:
        mua = b*(s*oxy+(1-s)*deoxy) + w*water + f*fat + m*melanin + c*collagen
    else:
        raise Exception("The sum of each substance's fraction is not equal to 1 !!")
    return mua


# %% main
# load for mua used
tissueMuaMcx = {tissue: [] for tissue in ["skin", "fat", "muscle", "ijv", "cca"]}
for wl in wlTarSet:
    with open(os.path.join(f"ijv_dis_op_avg_{wl}nm", f"mua_ijv_0.7_{wl}nm.json")) as f:
        muaMcx = json.load(f)
    for tidx, tissue in enumerate(["skin", "fat", "muscle", "ijv", "cca"]):
        tissueMuaMcx[tissue].append(muaMcx[tissueMcx[tidx]])

# load for mua cure
eachSubstanceMua = pd.read_csv(coefFile,
                               usecols = lambda x: "Unnamed" not in x)
wlp = eachSubstanceMua["wavelength"].values
oxy = eachSubstanceMua["oxy"].values
deoxy = eachSubstanceMua["deoxy"].values
water = eachSubstanceMua["water"].values
fat = eachSubstanceMua["fat"].values
melanin = eachSubstanceMua["mel"].values
collagen = eachSubstanceMua["collagen"].values
with open(tissueComTemPath) as f:
    tissueComTem = json.load(f)

# calculate mua curve
for tidx, tissue in enumerate(tissueSet):
    for widx, wl in enumerate(wlSet):
        mua = calculate_mua(tissueComTem[tissue]["BloodVol"], 
                               tissueComTem[tissue]["SO2"], 
                               tissueComTem[tissue]["WaterVol"], 
                               tissueComTem[tissue]["FatVol"], 
                               tissueComTem[tissue]["MelaninVol"], 
                               tissueComTem[tissue]["CollagenVol"], 
                               np.interp(wl, wlp, oxy), 
                               np.interp(wl, wlp, deoxy), 
                               np.interp(wl, wlp, water), 
                               np.interp(wl, wlp, fat), 
                               np.interp(wl, wlp, melanin), 
                               np.interp(wl, wlp, collagen)
                               )
        tissueMua[tidx][widx] = mua

# plot
# plt.figure(figsize=(2.8, 2.2))
mualns = []
for idx, tissue in enumerate(tissueSet):
    mualns += plt.plot(wlSet, tissueMua[idx], 
                       # label=tissue.title(),
                        label=tissue.upper()
                       )
    wlline = plt.plot(wlTarSet, np.array(tissueMuaMcx[tissue]) * 10, 
                      ".", color="red", 
                      label="730, 760, 780,\n810, 850 nm"
                      )
legend1 = plt.legend(wlline, [wlline[0].get_label()], edgecolor="white", 
                     fontsize="small", 
                      # bbox_to_anchor=[0.5, 0.64],
                       bbox_to_anchor=[0.43, 0.35],
                     )
plt.legend(mualns, [ln.get_label() for ln in mualns], 
           edgecolor="black", 
           fontsize="small")
plt.gca().add_artist(legend1)
plt.xlabel("Wavelength [nm]")
plt.ylabel("$\mu_a$ [$cm^{-1}$]")
plt.grid(visible=False)
# plt.title(tissue)
plt.show()




