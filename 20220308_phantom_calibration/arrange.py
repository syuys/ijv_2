#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 15:59:49 2022

@author: md703
"""

import pandas as pd
import json
import os
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


# function definition
def getMusp(wl, a, b):
    return a * (wl/765) ** (-b)  # 765 is just the mean point between [650, 881]

def getMus(musp, g=0):  # g = 0 for phantom (based on the past record)
    mus = musp/(1-g)
    return mus


# load setting
with open("phantom_test/config.json") as f:
    config = json.load(f)
    
with open("phantom_test/model_parameters.json") as f:
    modelParameters = json.load(f)
    
with open("phantom_test/mua.json") as f:
    mua = json.load(f)


# load optical properties of phantom
phantomMusp = pd.read_csv("musp_in_mm.csv")  # in mm
phantomMua = pd.read_csv("mua_in_mm.csv")    # in mm
phantomMusp = phantomMusp[phantomMusp["wl"] > 650]
phantomMusp = phantomMusp[phantomMusp["wl"] < 881]
phantomMua = phantomMua[phantomMua["wl"] > 650]
phantomMua = phantomMua[phantomMua["wl"] < 881]


# configure all session setting
phantomSet = "CHIK"
simWlSet = [730, 760, 810, 850]
phantomWl = phantomMusp["wl"].values
poptSet = {}

for phantom in phantomSet:
    # curve fit phantom musp
    poptSet[phantom], pcov = curve_fit(getMusp, phantomWl, phantomMusp[phantom].values)
    for wl in simWlSet:
        ### modify
        # config file
        session = f"{phantom}_{wl}"
        config["SessionID"] = session
        # model parameters file
        modelParameters["OptParam"]["Phantom body"]["mus"] = getMus(getMusp(wl, *poptSet[phantom]))
        # mua file
        mua["4: Phantom body"] = phantomMua[phantomMua["wl"] == wl][phantom].values[0]
        
        ### save setting
        # create folder
        os.mkdir(session)
        # save config
        with open(os.path.join(session, "config.json"), "w") as f:
            json.dump(config, f, indent=4)
        # save model parameters
        with open(os.path.join(session, "model_parameters.json"), "w") as f:
            json.dump(modelParameters, f, indent=4)
        # save mua
        with open(os.path.join(session, "mua.json"), "w") as f:
            json.dump(mua, f, indent=4)
        

