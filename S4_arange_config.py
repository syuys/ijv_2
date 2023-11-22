#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 23:49:55 2022

@author: md703
"""

import os
import json
from glob import glob

# %% parameters
projectID = "20230911_check_led_pattern_sdsrange_5to45_g99"
pathIDSet = glob(os.path.join(projectID, "ijv*"))  # *skin*fat*, ijv*, *EU*, ijv*_50%*
configPath = "config.json"
photonNum = 5e8
# stdtimesWay = "+0"  # eval( "pathID.split("_")[-2]" )
customizedCommands = ["--save2pt 1 --outputtype E --outputformat jnii"]

# %% load
if len(pathIDSet) == 0:
    raise Exception("Error in pathIDSet !")

# iteratively adjust config
for pathID in pathIDSet:
    sessionID = pathID.split("/")[-1]
    
    # para
    # ijv depth
    if sessionID.split("_")[2] == "depth":
        ijvDepthTimes = sessionID.split("_")[3]
    else:
        ijvDepthTimes = "+0"
    # cca sAng
    if sessionID.split("_")[2] == "ccasAng":
        ccasAng = sessionID.split("_")[3]
    # times = stdtimesWay
    # volTag = "_".join(pathID.split("_")[-3:])
    
    # adjust
    with open(os.path.join(pathID, configPath)) as f:
        config = json.load(f)
    config["PhotonNum"] = int(photonNum)
    # config["VolumePath"] = f"/home/md703/syu/ijv_2/ultrasound_image_processing/20230426_contrast_investigate_ijvdepth_sdsrange_5to45_g99/perturbed_tinyHolder_contrast_sds_5to45_ijv_depth_{ijvDepthTimes}_std.npy"
    # config["VolumePath"] = f"/home/md703/syu/ijv_2/ultrasound_image_processing/20230625_contrast_investigate_ijvaxis_sdsrange_5to45_g99/perturbed_tinyHolder_contrast_sds_5to45_ijv_{volTag}.npy"
    # config["VolumePath"] = f"/home/md703/syu/ijv_2/ultrasound_image_processing/20230712_contrast_investigate_cca_sAng_sdsrange_5to45_g99/perturbed_tinyHolder_contrast_sds_5to45_cca_sAng_{ccasAng}.npy"
    # config["VolumePath"] = "/home/md703/syu/ijv_2/ultrasound_image_processing/perturbed_tinyHolder_contrast_sds_5to45_ijv_minor_+3.5_mm.npy"
    config["VolumePath"] = "/home/md703/syu/ijv_2/ultrasound_image_processing/20230819_contrast_investigate_ijvpulserange_sdsrange_5to45_g99/perturbed_tinyHolder_contrast_sds_5to45_ijv_pulse_50%.npy"
    config["OutputPath"] = f"/home/md703/syu/ijv_2_output/{projectID}"
    config["CustomizedCommands"] = customizedCommands
    with open(os.path.join(pathID, configPath), "w") as f:
        json.dump(config, f, indent=4)
    
    print(f"sessionID: {sessionID}")