#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 04:32:37 2023

@author: md703
"""

from PyEMD import EMD
import numpy  as np
import matplotlib.pyplot as plt
plt.rcParams.update({"mathtext.default": "regular"})
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["figure.dpi"] = 300


#%% Define signal
s = signal["det_2"].mean(axis=1)
baseline = s[:40].mean()
t = np.arange(s.shape[0])

# Execute EMD on signal
IMF = EMD().emd(s, t)
N = IMF.shape[0]+1

# Plot results
fig, ax = plt.subplots(N, 1, figsize=(13, 8))
ax[0].plot(t, s, 'r')
ax[0].set_title("raw")

for n, imf in enumerate(IMF):
    ax[n+1].plot(t, imf, 'g')
    ax[n+1].set_title("imf " + str(n+1))
    
plt.xlabel("time [frame]")
plt.tight_layout()
plt.show()

# raw
plt.figure(figsize=(12, 4))
plt.plot(t, s, label="raw")
plt.plot(t, IMF[-1], label="potential detector shift")
plt.legend()
plt.title("raw")
plt.show()

# raw subtract shift
s = s - IMF[-1] + baseline
plt.figure(figsize=(12, 4))
plt.plot(t, s, label="raw - shift")
plt.plot(t, IMF[-3] + baseline, label="respiration")
plt.legend()
plt.title("raw subtract shift")
plt.show()

# raw subtract shift, respiration
s -= IMF[-3]
plt.figure(figsize=(12, 4))
plt.plot(t, s, label="raw - shift - respiration")
plt.plot(t, IMF[-2] + baseline, label="mayer wave")
plt.legend()
plt.title("raw subtract shift, respiration")
plt.show()

# raw subtract shift, respiration, mayer wave
s -= IMF[-2]
plt.figure(figsize=(12, 4))
plt.plot(t, s)
plt.title("raw - shift - respiration - mayer wave")
plt.show()