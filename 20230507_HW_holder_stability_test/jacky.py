# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 20:55:13 2021

@author: Hsin-Yuan
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PyEMD import EMD 
from scipy.signal import find_peaks_cwt, find_peaks, peak_prominences
plt.rcParams.update({"mathtext.default": "regular"})
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["figure.dpi"] = 300
#%%
def denoise(ori_spec):
    imfs = EMD().emd(ori_spec)
    artifact = imfs[-1] - imfs[-1].mean()
    spec_denoise = ori_spec - artifact
    return spec_denoise, artifact
#%%
path = r"D:\IJV\QEPro微型光譜儀控制介面\20210831\Test.csv"
data = pd.read_csv(path)
data = data.drop(['Wavelength (nm)'], axis=1)
wl = np.array(data.columns[0:].values, dtype=float) # wavelength
spec = np.array(data)  # raw data
wl_filter = (wl >= 725 ) & (wl <= 875)
wl = wl[wl_filter]
spec = spec[:, wl_filter] 
spec_time = spec.mean(axis=1) 

spec_time_denoise, artifact = denoise(spec_time)
#%%plot the intensity changes over time (reflectance v.s. time)
max_index = find_peaks_cwt(spec_time, np.arange(1, 12.5))
min_index = find_peaks_cwt(-spec_time, np.arange(1, 12.5))
    
max_index = find_peaks_cwt(spec_time_denoise, np.arange(1, 12))
min_index = find_peaks_cwt(-spec_time_denoise, np.arange(1, 12))


plt.figure()
plt.plot(spec_time)
plt.scatter(max_index, spec_time[max_index], label='max')
plt.scatter(min_index, spec_time[min_index], label='min')
plt.title("")
plt.legend()
plt.grid()
plt.xlabel("time[frame]")
plt.ylabel("reflectance[-]")

plt.figure()
plt.plot(spec_time_denoise)
plt.scatter(max_index, spec_time_denoise[max_index], label='max')
plt.scatter(min_index, spec_time_denoise[min_index], label='min')
plt.title("denosie")
plt.legend()
plt.grid()
plt.xlabel("time[frame]")
plt.ylabel("reflectance[-]")



#%%plot the max/min spectrum

# filter_max = peak_prominences(spec_time, max_index)[0] < 100
# filter_min = peak_prominences(-spec_time, min_index)[0] < 100
# max_index = max_index[filter_max]
# min_index = min_index[filter_min]


plt.figure()
live_max = (spec[max_index] - artifact[max_index].reshape(-1, 1)).mean(0)
live_min = (spec[min_index] - artifact[min_index].reshape(-1, 1)).mean(0)
# plt.xticks(wl)
plt.plot(wl, live_max, label="max")
plt.plot(wl, live_min, label="min")
plt.legend()

#%%
wavelength = np.arange(730, 911, 5)
R_max = np.empty(wavelength.shape, dtype=float)
R_min = np.empty(wavelength.shape, dtype=float)

for w, i in zip(wavelength, range(len(wavelength))):
    R_max[i]=(np.interp(w, wl, live_max)) 
    R_min[i]=(np.interp(w, wl, live_min))


plt.figure()
plt.plot(wavelength, R_max, label="max")
plt.plot(wavelength, R_min, label="min")
plt.legend()
#%%








