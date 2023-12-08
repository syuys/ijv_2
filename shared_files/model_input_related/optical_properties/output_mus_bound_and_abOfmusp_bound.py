#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 21:23:46 2021

@author: md703
"""

from IPython import get_ipython
get_ipython().magic('clear')
get_ipython().magic('reset -f')
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
plt.close("all")
import json
import os
from glob import glob
import scienceplots
plt.style.use(['science', 'grid'])
# plt.rcParams.update({"mathtext.default": "regular"})
# plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["figure.dpi"] = 300


# %% parameters and function
# parameters
tissueType = ["skin", "fat", "muscle", "blood"]
wlFiteredStart = 630
wlFilteredEnd = 1000
wlProjectStart = 725
wlProjectEnd = 875
muspBaseWl = (wlProjectStart+wlProjectEnd)/2
wlProject = np.linspace(wlProjectStart, wlProjectEnd, num=wlProjectEnd-wlProjectStart+1)
g_blood = 0.99
g_other = 0.9
issaveABbound  = False
issaveMUSbound = False

# function
def calculateMusp(wl, a, b):
    musp = a * (wl/muspBaseWl) ** (-b)
    return musp


# %% main
# present raw data and curve-fit data
musRange = {}
musRange["__comment__"] = "The mus upper bound and lower bound below are all in unit of [1/cm]."
abRange = {}
abRange["__comment__"] = "The (a, b) upper bound and lower bound below are all in unit of [1/cm]."
abRange["base wl"] = muspBaseWl
for tissue in tissueType:
    musp = {}
    muspPathSet = glob(os.path.join(tissue, "musp", "*.csv"))
    for muspPath in muspPathSet:
        # read values, select wavelength, and save
        name = muspPath.split("/")[-1].split(".")[0]
        df = pd.read_csv(muspPath)
        if tissue == "blood":
            df.musp = df.musp*10  # convert 1/mm to 1/cm when tissue's type is "blood".
        df = df[df.wavelength >= wlFiteredStart]
        df = df[df.wavelength <= wlFilteredEnd]
        musp[name] = {}
        musp[name]["values"] = df
        
        # curve fit
        popt, pcov = curve_fit(calculateMusp, df.wavelength.values, df.musp.values)
        musp[name]["(a,b)"] = popt
        
        # plot
        plt.plot(df.wavelength.values, df.musp.values, marker=".", label=name)
        plt.plot(np.linspace(df.wavelength.values[0], df.wavelength.values[-1], num=100), 
                 calculateMusp(np.linspace(df.wavelength.values[0], df.wavelength.values[-1], num=100), *popt), 
                 color=plt.gca().lines[-1].get_color(), linestyle="--", label=name+" - fit")
    plt.legend()
    plt.xlabel("wavelength [nm]")
    plt.ylabel("musp [1/cm]")
    plt.title(tissue + "'s musp")
    plt.show()
    
    # present curve-fit data only in project wavelength range and output largest and smallest musp, mus
    ab = np.empty((len(musp), 2))
    for idx, name in enumerate(musp.keys()):
        # rearrange (a,b)
        ab[idx] = musp[name]["(a,b)"]
        # plot curve-fit data
        plt.plot(wlProject, 
                 calculateMusp(wlProject, *ab[idx]), label=name+" - fit")
    if tissue == "blood":
        g = g_blood
    else:
        g = g_other
    # find smallest and largest musp, mus values
    bottomSteepestMusp = calculateMusp(wlProject, ab.min(axis=0)[0], ab.max(axis=0)[1])
    minMus = bottomSteepestMusp.min()/(1-g)
    print("Smallest {}'s musp: {:.4e}. (a,b) = ({:.3e}, {:.3e}). The mus: {}".format(tissue, bottomSteepestMusp.min(), 
                                                                                     ab.min(axis=0)[0], 
                                                                                     ab.max(axis=0)[1],
                                                                                     minMus))
    topSteepestMusp = calculateMusp(wlProject, ab.max(axis=0)[0], ab.max(axis=0)[1])
    maxMus = topSteepestMusp.max()/(1-g)
    print("Largest {}'s musp: {:.4e}. (a,b) = ({:.3e}, {:.3e}). The mus: {}".format(tissue, topSteepestMusp.max(), 
                                                                                    ab.max(axis=0)[0], 
                                                                                    ab.max(axis=0)[1],
                                                                                    maxMus), end="\n\n")
    # if tissue == "muscle":
    #     muscletopmusp = topSteepestMusp
    #     musclebotmusp = bottomSteepestMusp
    plt.plot(wlProject, bottomSteepestMusp, linestyle="--", color="gray")
    plt.plot(wlProject, topSteepestMusp, linestyle="--", color="gray")
    plt.legend()
    plt.xlabel("wavelength [nm]")
    plt.ylabel("musp [1/cm]")
    plt.title(tissue + "'s musp - only fit (project wl)")
    plt.show()
    
    # final setting - ab range & mus range
    abRange[tissue] = {}
    abRange[tissue]["a"] = [np.around(ab.min(axis=0)[0], 3), np.around(ab.max(axis=0)[0], 3)]
    bmin = ab[:, 1]
    bmin = bmin[bmin>0].min()
    abRange[tissue]["b"] = [np.around(bmin, 3), np.around(ab.max(axis=0)[1], 3)]
    musRange[tissue] = [np.floor(minMus), np.ceil(maxMus)]

# wltarget = [730, 760, 780, 810, 850]
# plt.plot(wlProject, muscletopmusp/0.1, label="Upper bound from literatures")
# plt.plot(wlProject, musclebotmusp/0.1, label="Lower bound from literatures")
# plt.legend(edgecolor="black", fontsize="small")
# plt.xlabel("Wavelength [nm]")
# plt.ylabel("$\mu_s$ [$cm^{-1}$]")
# plt.grid(visible=False)
# plt.title("Muscle")
# plt.show()

# save
if issaveABbound:
    with open("ab_bound_of_musp.json", "w") as f:
        json.dump(abRange, f, indent=4)
if issaveMUSbound:
    with open("mus_bound.json", "w") as f:
        json.dump(musRange, f, indent=4)
