#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 23:21:33 2022

@author: md703
"""

import numpy as np
import pandas as pd
import json
from glob import glob
import os
import matplotlib.pyplot as plt
plt.close("all")
plt.rcParams.update({"mathtext.default": "regular"})
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["figure.dpi"] = 300

# %% parameters and function
projectID = "20231212_contrast_invivo_geo_simulation_cca_pulse"
pathSet = glob(os.path.join(projectID, "ijv*"))  # *skin*fat*, *-1_std, *+0_std, ijv*, *EU*
muaSource = "from mua bound"  # "from mua bound" or "from coefficients"
# muaBoundPercentile = np.linspace(0.5, 0.5, 1)

#                               skin,   fat,muscle,   ijv,   cca
muaBoundPercentile = np.array([
                                # [   0,     0,     0,     0,     0],
                                # [0.25,  0.25,  0.25,  0.25,  0.25],
                                [ 0.5,   0.5,   0.5,   0.5,   0.5],
                                # [0.75,  0.75,  0.75,  0.75,  0.75],
                                # [   1,     1,     1,     1,     1],
                               ]
                              )

# muaBoundPercentile = np.array([[  0, 0.5, 0.5, 0.5, 0.5],
#                                [  1, 0.5, 0.5, 0.5, 0.5],
#                                [0.5,   0, 0.5, 0.5, 0.5],
#                                [0.5,   1, 0.5, 0.5, 0.5],
#                                [0.5, 0.5,   0, 0.5, 0.5],
#                                [0.5, 0.5,   1, 0.5, 0.5],
#                                [0.5, 0.5, 0.5,   0,   0],
#                                [0.5, 0.5, 0.5,   1,   1],
#                                [0.5, 0.5, 0.5,   0, 0.5],
#                                [0.5, 0.5, 0.5,   1, 0.5],
#                                [0.5, 0.5, 0.5, 0.5,   0],
#                                [0.5, 0.5, 0.5, 0.5,   1],
#                                ]
#                               )

tissueComTemTag = "tissue_composition_template_ijv_0.7.json"
tissueSet = ["4: Skin", "5: Fat", "6: Muscle", "8: IJV", "9: CCA"]
isSave = True

def calculate_mua(b, s, w, f, m, c, oxy, deoxy, water, fat, melanin, collagen):
    """
    Calculate mua according to volume fraction.
    The sum of each substance's fraction should be 1.

    Parameters
    ----------
    b : blood fraction
    s : So2    
    w : water fraction
    f : fat fraction
    m : melanin fraction
    c : collagen fraction
    oxy : HbO2 mua  [1/mm]
    deoxy : Hb mua  [1/mm]
    water : water mua  [1/mm]
    fat : fat mua  [1/mm]
    melanin : melanin mua  [1/mm]
    collagen : collagen mua  [1/mm]

    Returns
    -------
    mua : mua  [1/mm]

    """
    if b+w+f+m+c == 1:
        mua = b*(s*oxy+(1-s)*deoxy) + w*water + f*fat + m*melanin + c*collagen
    else:
        raise Exception("The sum of each substance's fraction is not equal to 1 !!")
    return mua

# %% arange
if muaSource == "from mua bound":
    with open("shared_files/model_input_related/optical_properties/mua_bound.json") as f:
        muabound = json.load(f)
    with open(os.path.join(projectID, "mua_template.json")) as f:
        mua = json.load(f)
    
    for path in pathSet:
        ijvType = path.split("/")[-1].split("_")[1]
        for p in muaBoundPercentile:
            print(f"========= {os.path.split(path)[-1]},  {p}  =========")
            for idx, tissue in enumerate(tissueSet):
                target = tissue.split(" ")[-1].lower()
                bound = muabound[target]
                select = bound[0] + (bound[1]-bound[0]) * p[idx]
                select = np.around(select/10, 4)  # cm -> mm
                
                # assign calculated mua
                mua[tissue] = select
                if (target == "muscle") & (ijvType == "col"):
                    mua["7: Muscle or IJV (Perturbed Region)"] = select
                elif (target == "ijv") & (ijvType == "dis"):
                    mua["7: Muscle or IJV (Perturbed Region)"] = select
                
                print(f"{target} bound: {bound}  # 1/cm")
                print(f"choose: {select}  # 1/mm")
                print()
            tissuemua = f"skin_{int(p[0]*100)}%_fat_{int(p[1]*100)}%_muscle_{int(p[2]*100)}%_ijv_{int(p[3]*100)}%_cca_{int(p[4]*100)}%"
            if isSave:
                with open(os.path.join(path, f"mua_{tissuemua}.json"), "w") as f:
                    json.dump(mua, f, indent=4)

elif muaSource == "from coefficients":
    # load
    eachSubstanceMua = pd.read_csv("shared_files/model_input_related/optical_properties/coefficients_in_cm-1_OxyDeoxyConc150.csv",
                                   usecols = lambda x: "Unnamed" not in x)
    wlp = eachSubstanceMua["wavelength"].values
    tissueComTemPath = glob(os.path.join(projectID, tissueComTemTag))
    tissueComTemPath.sort(key=lambda x: float(x[-8:-5]))
    tissueComTemSet = []
    for path in tissueComTemPath:
        with open(path) as f:
            tissueComTemSet.append(json.load(f))
    # with open(os.path.join(projectID, "tissue_composition_template.json")) as f:
    #     tissueComTem = json.load(f)
    with open(os.path.join(projectID, "mua_template.json")) as f:
        mua = json.load(f)
    
    # mua ref
    tissueref = ["skin", "fat", "muscle", "ijv", "cca"]
    muaref = {}
    for tissue in tissueref:
        muaref[tissue] = {}
        refPathSet = glob(os.path.join("shared_files/model_input_related/optical_properties/all_tissue_interp_mua", tissue, "*"))
        for path in refPathSet:
            data = pd.read_csv(path)
            data = np.array(data)
            muaref[tissue][path.split("/")[-1]] = data    
    
    # unit conversion  1/cm -> 1/mm
    oxy = eachSubstanceMua["oxy"].values * 0.1
    deoxy = eachSubstanceMua["deoxy"].values * 0.1
    water = eachSubstanceMua["water"].values * 0.1
    fat = eachSubstanceMua["fat"].values * 0.1
    melanin = eachSubstanceMua["mel"].values * 0.1
    collagen = eachSubstanceMua["collagen"].values * 0.1    
    
    pathSet.sort(key=lambda x: (int(x[-5:-2]), -ord(os.path.split(x)[-1][4])))
    for tissueComTem in tissueComTemSet:
        # mua setup
        tissueMua = {f"{tissue}":[] for tissue in tissueSet}
        wlSet = []
        ijvso2 = f"ijv_{tissueComTem['ijv']['SO2']}"
        for path in pathSet:
            wl = int(path[-5:-2])
            ijvType = path.split("/")[-1].split("_")[1]
            if ijvType == "dis": wlSet.append(wl)
            
            print(f"\n(wl, ijvType): ({wl}, {ijvType}), {ijvso2}")
            for tissue in tissueSet:
                target = tissue.split(" ")[-1].lower()
                print(f"target: {target}")
                select = calculate_mua(tissueComTem[target]["BloodVol"], 
                                       tissueComTem[target]["SO2"], 
                                       tissueComTem[target]["WaterVol"], 
                                       tissueComTem[target]["FatVol"], 
                                       tissueComTem[target]["MelaninVol"], 
                                       tissueComTem[target]["CollagenVol"], 
                                       np.interp(wl, wlp, oxy), 
                                       np.interp(wl, wlp, deoxy), 
                                       np.interp(wl, wlp, water), 
                                       np.interp(wl, wlp, fat), 
                                       np.interp(wl, wlp, melanin), 
                                       np.interp(wl, wlp, collagen)
                                       )
                mua[tissue] = select
                if ijvType == "dis": tissueMua[tissue].append(select)
                
                if (target == "muscle") & (ijvType == "col"):
                    mua["7: Muscle or IJV (Perturbed Region)"] = select
                    print(f"--- select: {select} ---")
                elif (target == "ijv") & (ijvType == "dis"):
                    mua["7: Muscle or IJV (Perturbed Region)"] = select
                    print(f"--- select: {select} ---")
                    
            # save mua to the path
            if isSave:
                with open(os.path.join(path, f"mua_{ijvso2}_{wl}nm.json"), "w") as f:
                    json.dump(mua, f, indent=4)
        
        # plot each tissue final mua
        for key, value in tissueMua.items():
            # compared to reference range
            for ref in muaref[key.split(" ")[-1].lower()].values():
                for data in np.transpose(ref)[1:]:
                    plt.plot(np.transpose(ref)[0], data, color='#1f77b4')
            # used mua
            plt.plot(wlSet, np.array(value)*10, "-o", color="red")
            
            plt.xlabel("wl [nm]")
            plt.ylabel("mua [1/cm]")
            plt.title(f"{ijvso2}, {key} mua")
            plt.show()

else:
    raise Exception("muaSource type is not defined !")

