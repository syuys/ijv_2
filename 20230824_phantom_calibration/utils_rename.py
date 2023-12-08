import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl  


def moving_avg(used_wl_bandwidth, used_wl, time_mean_arr):
    ''' 
    used_wl_bandwidth --> nm the range want to average
    used_wl --> 1D array
    time_mean_arr --> 1D array (spectrum) correspond with used_wl
    '''
    resolution = used_wl[1] - used_wl[0]
    average_points = int(used_wl_bandwidth/resolution)
    moving_avg_I = []
    moving_avg_wl = []
    for i in range(time_mean_arr.shape[0]-average_points):
        moving_avg_I += [time_mean_arr[i:i+average_points].mean()]
        if average_points % 2 == 1: # odd 
            moving_avg_wl += [used_wl[i+average_points//2]]
        else: # even
            even = (used_wl[i+average_points//2] + used_wl[i+average_points//2-1]) * 0.5
            moving_avg_wl += [even]
    
    return np.array(moving_avg_I), np.array(moving_avg_wl)

def plot_individual_phantom(used_wl, time_mean_np_arr, time_mean_np_arr_remove_bg, moving_avg_I, moving_avg_wl, name, savepath):
    plt.plot(used_wl, time_mean_np_arr, label='raw data')
    plt.title(f'phantom_{name} spectrum')
    plt.xlabel('wavelength (nm)')
    plt.ylabel('intensity')
    plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5),
                    fancybox=True, shadow=True)
    plt.savefig(os.path.join("pic", savepath, "raw_data.png"), dpi=300, format='png', bbox_inches='tight')
    plt.show()
    
    plt.plot(used_wl, time_mean_np_arr_remove_bg, label='raw data - bg')
    plt.title(f'phantom_{name} spectrum')
    plt.xlabel('wavelength (nm)')
    plt.ylabel('intensity')
    plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5),
                    fancybox=True, shadow=True)
    plt.savefig(os.path.join("pic", savepath, "raw_data_remove_bg.png"), dpi=300, format='png', bbox_inches='tight')
    plt.show()
    
    plt.plot(moving_avg_wl, moving_avg_I, label='moving average')
    plt.title(f'phantom_{name} spectrum')
    plt.xlabel('wavelength (nm)')
    plt.ylabel('intensity')
    plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5),
                    fancybox=True, shadow=True)
    plt.savefig(os.path.join("pic", savepath, "moving_avg.png"), dpi=300, format='png', bbox_inches='tight')
    plt.show()

def plot_each_time_and_remove_spike(used_wl, np_arr, name, savepath):
    for data in np_arr:
        plt.plot(used_wl, data)
    plt.title(f'phantom_{name} spectrum')
    plt.xlabel('wavelength (nm)')
    plt.ylabel('intensity')
    plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5),
                    fancybox=True, shadow=True)
    plt.savefig(os.path.join("pic", savepath, "each_time_raw_data.png"), dpi=300, format='png', bbox_inches='tight')
    plt.show()
    
    remove_spike_data = remove_spike(used_wl, np_arr, name, normalStdTimes=10, showTargetSpec=True, savepath=savepath)
    
    for data in remove_spike_data:
        plt.plot(used_wl, data)
    plt.title(f'phantom_{name} spectrum')
    plt.xlabel('wavelength (nm)')
    plt.ylabel('intensity')
    plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5),
                    fancybox=True, shadow=True)
    plt.savefig(os.path.join("pic", savepath, "each_time_raw_data_remove_spike.png"), dpi=300, format='png', bbox_inches='tight')
    plt.show()
    
    return remove_spike_data

def remove_spike(used_wl, data, normalStdTimes, showTargetSpec):
    mean = data.mean(axis=0)
    std = data.std(ddof=1, axis=0)
    targetSet = []  # save spike idx
    for idx, s in enumerate(data):  # iterate spectrum in every time frame
        isSpike = np.any(abs(s-mean) > normalStdTimes*std)
        if isSpike:
            targetSet.append(idx) 
    print(f"target = {targetSet}")
    if len(targetSet) != 0:
        for target in targetSet:
            # show target spec and replace that spec by using average of the two adjacents
            if showTargetSpec:
                plt.plot(used_wl, data[target])
                plt.xlabel("Wavelength [nm]")
                plt.ylabel("Intensity [counts]")    
                plt.title(f"spike idx: {target}")
                plt.savefig(os.path.join("pic", savepath, f"spike_at_time_stamp_{target}.png"), dpi=300, format='png', bbox_inches='tight')
                plt.show()
            
            data[target] = (data[target-1] + data[target+1]) / 2
            
    return data

def cal_R_square(y_true, y_pred):
    y_bar = np.mean(y_true)
    numerator = np.sum(np.square(y_true-y_pred))
    denominator = np.sum(np.square(y_true-y_bar))
    R_square = 1 - numerator/denominator
    
    return R_square