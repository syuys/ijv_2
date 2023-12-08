#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 17:37:16 2023

@author: md703
"""

import numpy as np
import os
import pandas as pd
import json
import matplotlib.pyplot as plt
import matplotlib as mpl               
from utils import moving_avg, remove_spike, plot_individual_phantom, plot_each_time_and_remove_spike, cal_R_square
# plot style format                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
mpl.rcParams.update(mpl.rcParamsDefault)
plt.style.use("seaborn-darkgrid")
plt.rcParams.update({"mathtext.default": "regular"})
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["figure.dpi"] = 300


# %% setting format
date = "20230812"
phantom_measured_ID = ['2', '3', '4', '5']
SDS_idx = 4  # SDS=10 mm
SDS = 20 # [mm]


# %% Load measured phantom data
# get background Nx : time, Ny : intensity
background = pd.read_csv(os.path.join("dataset", date, "background.csv"))
used_wl = []
for k in background.keys().to_list()[1:]:
    used_wl += [float(k)]
background = background.to_numpy()[:,1:]
background = np.array(background, dtype=np.float64)

remove_spike_background = remove_spike(used_wl, background, normalStdTimes=10, showTargetSpec=False) # remove spike
time_mean_background = remove_spike_background.mean(axis=0) # mean of background signal

# get measured phantom data
phantom_data = [] # CHIK3456
for ID in phantom_measured_ID:
    # define plot savepath
    savepath = os.path.join(date, ID)
    os.makedirs(os.path.join("pic", savepath), exist_ok=True)
    
    # import measured data
    data = pd.read_csv(os.path.join('dataset', date, f'{ID}.csv'))
    data = data.to_numpy()[:,1:]
    data = np.array(data, dtype=np.float64)
    
    # remove spike
    remove_spike_data = remove_spike(used_wl, data, normalStdTimes=10, showTargetSpec=False) # remove spike
    time_mean_data = remove_spike_data.mean(0) # mean of measured signal
    
    # subtract background
    time_mean_data_sub_background = time_mean_data - time_mean_background
    
    # Do moving avg of spectrum
    moving_avg_I_data, moving_avg_wl_data = moving_avg(used_wl_bandwidth = 3,
                                                       used_wl = used_wl,
                                                       time_mean_arr = time_mean_data_sub_background)
    
    phantom_data.append(moving_avg_I_data)


# %% Load simulated phantom data
# load used wavelength
with open(os.path.join("OPs_used", "wavelength.json"), "r") as f:
    wavelength = json.load(f)
    wavelength = wavelength['wavelength']

# binning meaurement wavelength 
binning_wavlength = 2 # nm
find_wl_idx = {}
for used_wl in wavelength:
    row = []
    for idx, test_wl in enumerate(moving_avg_wl_data):
        if abs(test_wl-used_wl) < binning_wavlength:
            row += [idx]          
    find_wl_idx[used_wl] = row


## Get the same simulated wavelength point from measured phantom
measured_phantom_data = []
used_phantom_data = phantom_data
for idx, data in enumerate(used_phantom_data):
    avg_wl_as_data = []
    for k in find_wl_idx.keys():
        average_idx = find_wl_idx[k]
        avg_wl_as_data += [data[average_idx].mean()]
    measured_phantom_data.append(avg_wl_as_data)
    plt.plot(wavelength, avg_wl_as_data, 'o--', label=f'phantom_{phantom_measured_ID[idx]}')
plt.legend()
plt.savefig(os.path.join("pic", date, "measured_phantom_adjust_wl.png"), dpi=300, format='png', bbox_inches='tight')
plt.title("measured result")
plt.show()
measured_phantom_data = np.array(measured_phantom_data)


## Get simulated phantom data
sim_phantom_data = []
for c in phantom_measured_ID:
    data = np.load(os.path.join("dataset", "phantom_simulated", f'{c}.npy'))
    sim_phantom_data.append(data[:,SDS_idx].tolist())
    plt.plot(wavelength, data[:,SDS_idx], 'o--',label=f'phantom_{c}')
plt.title("simulation result")
plt.legend()
plt.savefig(os.path.join("pic", date, "simulated_phantom_adjust_wl.png"), dpi=300, format='png', bbox_inches='tight')
plt.show()
sim_phantom_data = np.array(sim_phantom_data)


# %% Fit measured phantom and simulated phantom
# delete first 3 wl data
wavelength = wavelength[3:]
measured_phantom_data = measured_phantom_data[:, 3:]
sim_phantom_data = sim_phantom_data[:, 3:]

fig = plt.figure(figsize=(18,12))
fig.suptitle(f"SDS = {SDS} mm", fontsize=16)
count = 1
for idx, used_wl in enumerate(wavelength):
    ## fit measured phantom and simulated phantom
    z = np.polyfit(measured_phantom_data[:, idx], sim_phantom_data[:,idx], 1)
    plotx = np.linspace(measured_phantom_data[-1, idx]*0.8,  measured_phantom_data[0, idx]*1.2,100)
    ploty = plotx*z[0] + z[1]
    calibrate_data = measured_phantom_data[:, idx]*z[0] + z[1]
    R_square = cal_R_square(calibrate_data, sim_phantom_data[:,idx]) # cal R square
    
    ## plot result
    ax = plt.subplot(5,4, count)
    ax.set_title(f"@wavelength={used_wl} nm")
    ax.set_title(f'{used_wl}nm, $R^{2}$={R_square:.2f}')
    for ID_idx, ID in enumerate(phantom_measured_ID):
        ax.plot(measured_phantom_data[ID_idx, idx], sim_phantom_data[ID_idx,idx], 's', label=f'phantom_{ID}')
    ax.plot(plotx, ploty, '--')
    ax.set_xlabel("sim intensity")
    ax.set_ylabel("measure intensity")
    count += 1
plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5),
                    fancybox=True, shadow=True)
plt.tight_layout()
plt.savefig(os.path.join('pic', savepath, "all.png"), dpi=300, format='png', bbox_inches='tight')
plt.show()