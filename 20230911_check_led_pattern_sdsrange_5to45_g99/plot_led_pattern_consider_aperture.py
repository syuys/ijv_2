#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 21:43:07 2023

@author: md703
"""

from postprocess import plotIntstDistrb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["figure.dpi"] = 300


# %% parameters
projectID = "20230911_check_led_pattern_sdsrange_5to45_g99"
sessionID = "ijv_dis_pulse_50%"
voxelSize = 0.25
top = "top.npy"

# %% main
ledcdf = pd.read_csv("/home/md703/syu/ijv_2/shared_files/model_input_related/LED_profile_in3D_pfForm_0to89.csv")
ledcdf = ledcdf.to_numpy()
top = np.load(top)

xyDistrbNearSource = plotIntstDistrb(projectID, sessionID)

# plt.plot(ledcdf[:, 0], np.cumsum(ledcdf[:, 1]))
plt.plot(ledcdf[:, 0], ledcdf[:, 1])
plt.show()

# %% radial average and plot
line = np.arange(0, 2.5, 3/100)
theta = np.arange(0, 2*np.pi, 2*np.pi/1000)
x = line * np.cos(theta)[:, None]
y = line * np.sin(theta)[:, None]
x = np.floor(x/voxelSize).astype(int)
y = np.floor(y/voxelSize).astype(int)
x += xyDistrbNearSource.shape[0] // 2
y += xyDistrbNearSource.shape[1] // 2
radialavg = xyDistrbNearSource[x, y]
radialavg = radialavg.mean(axis=0)
plt.plot(line, radialavg, marker=".", label="Radial average")

# plot intensity trend in the first skin layer near source
hori_x = np.linspace(0.125, 2.375, 10)
hori_in = (xyDistrbNearSource[xyDistrbNearSource.shape[0]//2-1] + xyDistrbNearSource[xyDistrbNearSource.shape[0]//2]) / 2
plt.plot(hori_x, hori_in[10:], marker=".", label="Horizon")
# plt.xticks(np.linspace(-0.5, xyDistrbNearSource.shape[0]-0.5, num=5), 
#            np.linspace(-xyDistrbNearSource.shape[0]*voxelSize/2, xyDistrbNearSource.shape[0]*voxelSize/2, num=5))
plt.xlabel("Distance from center [mm]")
plt.ylabel("Energy density [-]")
plt.legend()
plt.title("First skin layer")
# plt.title("Distribution of normalized intensity in the first skin layer near source")
plt.show()





