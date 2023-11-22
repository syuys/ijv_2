#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 23:09:47 2022

@author: md703
"""

import numpy as np
import json
from glob import glob
import os

# %% parameters
projectID = "20230911_check_led_pattern_sdsrange_5to45_g99"
pathSet = glob(os.path.join(projectID, "ijv*"))  # ijv*, *skin_100%*fat*, *skin_0%*fat*, *EU*_100%*
modelParamPath = "model_parameters.json"
musSource = "from mus bound"  # "from ab bound" or "from mus bound"
percentileMUSbound = 0.5  # 0.5, "float(path.split('_')[-1][:-1]) / 100"
percentileABbound = 0.5
g_blood = 0.99
g_other = 0.9


# %% arange
if musSource == "from mus bound":
    with open("shared_files/model_input_related/optical_properties/mus_bound.json") as f:
        musBound = json.load(f)

    for path in pathSet:
        
        # percentileType = (path.split('_')[-2], eval(percentileMUSbound))
        print(f"\npath: {path}")
        # print(f"Percentile: {percentileType}")
        
        with open(os.path.join(path, modelParamPath)) as f:
            modelParam = json.load(f)
            
        for tissue in ["skin", "fat", "muscle", "blood"]:
            # percentile = percentileType[1] if tissue == percentileType[0] else 0.5
            percentile = percentileMUSbound
            print(f"(tissue, percentile) = ({tissue}, {percentile})")
            
            bound = musBound[tissue]
            select = bound[0] + (bound[1]-bound[0]) * percentile
            select = np.around(select/10, 3)  # cm -> mm
            print(f"{tissue} bound: {bound}  # 1/cm")
            print(f"choose: {select}  # 1/mm")
            
            if tissue == "blood":
                modelParam["OptParam"]["IJV"]["mus"] = select
                modelParam["OptParam"]["IJV"]["g"] = g_blood
                modelParam["OptParam"]["CCA"]["mus"] = select
                modelParam["OptParam"]["CCA"]["g"] = g_blood
            else:
                modelParam["OptParam"][tissue.title()]["mus"] = select
                modelParam["OptParam"][tissue.title()]["g"] = g_other
    
        with open(os.path.join(path, modelParamPath), "w") as f:
            json.dump(modelParam, f, indent=4)

elif musSource == "from ab bound":
    with open("shared_files/model_input_related/optical_properties/ab_bound_of_musp.json") as f:
        abBound = json.load(f)
    muspBaseWl = abBound["base wl"]
    for path in pathSet:
        percentile = percentileABbound
        wl = int(path[-5:-2])
        print(f"\nWavelength: {wl}")
        with open(os.path.join(path, modelParamPath)) as f:
            modelParam = json.load(f)
        for tissue in ["skin", "fat", "muscle", "blood"]:
            bound = abBound[tissue]
            a = bound["a"][0] + (bound["a"][1]-bound["a"][0]) * percentile
            b = bound["b"][0] + (bound["b"][1]-bound["b"][0]) * percentile
            musp = a * (wl/muspBaseWl) ** (-b)
            musp /= 10  # cm -> mm
            if tissue == "blood":
                mus = musp / (1-g_blood)
                modelParam["OptParam"]["IJV"]["g"] = g_blood  # 20230319 g: 0.95 -> 0.99
                modelParam["OptParam"]["CCA"]["g"] = g_blood
                modelParam["OptParam"]["IJV"]["mus"] = mus
                modelParam["OptParam"]["CCA"]["mus"] = mus
            else:
                mus = musp / (1-g_other)
                modelParam["OptParam"][tissue.title()]["g"] = g_other
                modelParam["OptParam"][tissue.title()]["mus"] = mus
            print(f"{tissue} ab bound: {bound}  # 1/cm")
            print(f"ab : ({a}, {b})")
            print(f"mus: {mus}  # 1/mm")
        
        with open(os.path.join(path, modelParamPath), "w") as f:
            json.dump(modelParam, f, indent=4)

else:
    raise Exception("musSource type is not defined !")


