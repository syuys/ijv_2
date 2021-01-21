#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 11:35:19 2020

@author: md703
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import json
plt.rcParams.update({'mathtext.default': 'regular'})
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['figure.dpi'] = 150

#%% function
ref_wl_jacques = 500
def calculate_musp_jacques(wl, a, b):
    musp = a * (wl/ref_wl_jacques) ** (-b)
    return musp

ref_wl_ijv = 745 # middle point of wl 680~810
def calculate_musp_ijv(wl, a, b):
    musp = a * (wl/ref_wl_ijv) ** (-b)
    return musp

#%% basic parameters

# our project's wl
wl = np.linspace(680, 810, num=14, dtype=int)

# input path - the file of literature datas
input_path = "input/musp_a_b_literatureValues.json"

# output path - to save the transformed (a,b) range into a stable file
output_path = "input/musp_a_b_range_ijvValues.json"

#%% calculate and plot literature's musp, in order to find the bound in previous studies

with open(input_path) as f:
    para_set_jacques = json.load(f)["contents"] # blood added on 12/4, for convenience. (not from Jacques, is from toast's estimation)

# musp result container (waiting for the values calculated from literatures)
musp_info = {}

# iterate over different tissues
for tissue in para_set_jacques.keys():
    
    musp_info[tissue] = {}
    musp_info[tissue]["musp"] = np.empty((0, wl.size), dtype=float)
    
    for ref, para in para_set_jacques[tissue].items():
        
        # calculate musp using jacques's equation and save to result container
        musp = calculate_musp_jacques(wl, *para)
        musp_info[tissue]["musp"] = np.append(musp_info[tissue]["musp"], musp[None, :], axis=0)
        
        # plot every musp of the tissue
        plt.plot(wl, musp, '--.', label="{}: \na = {}, b = {}".format(ref, *para))
    
    # find the lower bound, upper bound of this tissue musp and save
    musp_info[tissue]["lower bound"] = musp_info[tissue]["musp"].min(axis=0)
    musp_info[tissue]["upper bound"] = musp_info[tissue]["musp"].max(axis=0)
    
    # mark the lower bound and upper bound on the figure
    plt.plot(wl, musp_info[tissue]["lower bound"], label="lower bound")
    plt.plot(wl, musp_info[tissue]["upper bound"], label="upper bound") 
    plt.grid()
    plt.legend(bbox_to_anchor=(1, 1.023))
    plt.xlabel("Wavelength [nm]", fontsize=12)
    plt.ylabel("$\mu_s\prime$ [cm$^{-1}$]", fontsize=12)
    plt.title(r"{}, $\mu_s\prime$ = a $\bullet$ ".format(str.capitalize(tissue)) + "($\lambda$/{})".format(
        ref_wl_jacques) + "$^{-b}$ (Jacques)", fontsize=12)
    # plt.savefig("ref musp of {}.png".format(tissue), dpi=300, bbox_inches='tight')
    plt.show()

#%% ===================== OUR PROJECT ==========================
#%% covert jacques's (a,b) to the (a,b) of our ijv project ('b' will be the same, but 'a' will be changed.)

para_set_ijv = {}

# keys are from para_set_jacques
for tissue in para_set_jacques.keys():
    
    para_set_ijv[tissue] = {}
    
    # materials of ref and para are from para_set_jacques[tissue]
    for ref, para in para_set_jacques[tissue].items():
        
        # Calculate the musp values based on Jacques's equation first
        musp = calculate_musp_jacques(wl, *para)
        
        # And then do curve fitting to make 'a' be our musp_745nm (ref_wl_ijv)
        popt, pcov = curve_fit(calculate_musp_ijv, wl, musp)
        
        # save the new (a,b) coefficients to para_set_ijv
        para_set_ijv[tissue][ref] = tuple(popt)

#%% plot musp lowerbound and upperbound of previous studies, and plot all musp uniformly derived from the lb & ub of para_set_ijv (a,b),
### in order to check whether the musps drawn from transformed (a,b) are distributed reasonable. (some oscillation is accepted
### on the boundary)
### Additionally, save the lb and ub of the transformed (a,b) values.

a_b_range_forijv = {}

for tissue in musp_info.keys():
    # find lb and ub of (a,b) from para_set_ijv
    a_lb = np.array(list(para_set_ijv[tissue].values())).min(axis=0)[0]
    a_ub = np.array(list(para_set_ijv[tissue].values())).max(axis=0)[0]
    b_lb = np.array(list(para_set_ijv[tissue].values())).min(axis=0)[-1]
    b_ub = np.array(list(para_set_ijv[tissue].values())).max(axis=0)[-1]
    
    # save lb and ub
    a_b_range_forijv[tissue] = {}
    a_b_range_forijv[tissue]["a"] = [a_lb, a_ub]
    a_b_range_forijv[tissue]["b"] = [b_lb, b_ub]
    
    # slice uniformly
    a_set = np.linspace(a_lb, a_ub, num=4)
    b_set = np.linspace(b_lb, b_ub, num=4)
    
    # print each tissue's a_range & b_range
    print("{}: \na_range = {}, \nb_range = {}\n".format(tissue, [a_lb, a_ub], [b_lb, b_ub]))
    
    # plot
    fig, ax = plt.subplots()
    
    ax.plot(wl, musp_info[tissue]["lower bound"], label='lower bound')
    ax.plot(wl, musp_info[tissue]["upper bound"], label='upper bound')
    
    for b in b_set:
        for a in a_set:
            ax.plot(wl, calculate_musp_ijv(wl, a, b), color="gray", linestyle="dashed")
            
    ax.text(0.67, 0.93, "a_range = [{:.3f}, {:.3f}]\nb_range = [{:.3f}, {:.3f}]".format(a_lb, a_ub, b_lb, b_ub),
             horizontalalignment="right", verticalalignment="center",
             transform=ax.transAxes)
    
    # ax.grid()
    ax.legend()
    ax.set_xlabel("Wavelength [nm]", fontsize=12)
    ax.set_ylabel("$\mu_s\prime$ [cm$^{-1}$]", fontsize=12)
    ax.set_title(r"{} $\mu_s\prime$ sampling from uniform (a,b) distribution, $\mu_s\prime$ = a $\bullet$ ($\lambda$/{})".format(
        str.capitalize(tissue), ref_wl_ijv) + "$^{-b}$", fontsize=12)
    plt.show()
    # fig.savefig("musp of {} drawn from uniform (a,b) distribution (ijv).png".format(tissue), dpi=300, bbox_inches='tight')
    
#%% output final (a,b) range for ijv's use

comment = "The (a,b) range below is transformed from 'musp_a_b_literatureValues.json'. \
Actually, 'a' is changed, and 'b' keeps the same. The equation for musp calculation \
is --> musp = a * (wl/745)^-b, which 745nm is the reference wl of our ijv project."

final_a_b_range_forijv = {"//comment": comment,
                          "contents": a_b_range_forijv}

with open(output_path, "w") as f:
    json.dump(final_a_b_range_forijv, f, indent=4)
