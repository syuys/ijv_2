#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 23:08:15 2023

@author: md703
"""

import os
from glob import glob
import socket
from datetime import datetime
import time


# parameters
projectID = "20231212_contrast_invivo_geo_simulation_cca_pulse"
projectPath = f"/home/md703/syu/ijv_2_output/{projectID}"
sessionPathSet = glob(os.path.join(projectID, "ijv*"))
host = "md703@192.168.31.104"  # original: md703@192.168.31.237
hostPathData = f"/media/md703/Expansion/syu/ijv_2_output/{projectID}"
hostPathInfo = f"/home/md703/syu/ijv_2_output/{projectID}"
fileExtent = "*.jdat"
orderPos = -2


# get current ip and host ip
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.connect(("8.8.8.8", 80))
ip_now = s.getsockname()[0]
s.close()
ip_host = host.split("@")[-1]

# check if project, session, and mcxOutput folder in host path has been created
command = f"sshpass -p md703 ssh {host} 'if [ ! -d {hostPathData} ]; then mkdir {hostPathData}; fi'"
os.system(command)
command = f"sshpass -p md703 ssh {host} 'if [ ! -d {hostPathInfo} ]; then mkdir {hostPathInfo}; fi'"
os.system(command)
for sessionPath in sessionPathSet:
    sessionID = sessionPath.split("/")[-1]
    
    sessionHostPathData = os.path.join(hostPathData, sessionID)
    command = f"sshpass -p md703 ssh {host} 'if [ ! -d {sessionHostPathData} ]; then mkdir {sessionHostPathData}; fi'"
    os.system(command)
    
    mcxOutputHostPathData = os.path.join(sessionHostPathData, "mcx_output")
    command = f"sshpass -p md703 ssh {host} 'if [ ! -d {mcxOutputHostPathData} ]; then mkdir {mcxOutputHostPathData}; fi'"
    os.system(command)
    
    sessionHostPathInfo = os.path.join(hostPathInfo, sessionID)
    command = f"sshpass -p md703 ssh {host} 'if [ ! -d {sessionHostPathInfo} ]; then mkdir {sessionHostPathInfo}; fi'"
    os.system(command)


# start sending task
print(f"Project: {projectID}")
print("Checking...")
while True:    
    for sessionPath in sessionPathSet:
        sessionID = sessionPath.split("/")[-1]
        detOutputPathSet = glob(os.path.join(projectPath, sessionID, "mcx_output", "backup", fileExtent))
        if len(detOutputPathSet) >= 10:
            startTime = time.time()
            # order = [int(i.split("_")[orderPos].split(".")[0]) for i in detOutputPathSet]
            order = [int(i.split("_")[orderPos]) for i in detOutputPathSet]
            # show sending info
            currentTime = datetime.now().strftime("%m/%d %H:%M:%S")
            print(f"{currentTime},  {sessionID},  sending {min(order)} ~ {max(order)},  ", end="")
            
            sessionHostPathData = os.path.join(hostPathData, sessionID)
            
            # send data
            jdatHostPathData = os.path.join(sessionHostPathData, 'mcx_output')
            for detOutputPath in detOutputPathSet:
                command = f"sshpass -p md703 scp -r {detOutputPath} {host}:{jdatHostPathData}"
                state = os.system(command)
                if state == 0:
                    os.remove(detOutputPath)
                else:
                    print(f"Error state = {state}")
                    raise Exception(f"Sending error !! → {detOutputPath}")
            
            # send other folders
            sendFolderSet = ["json_output", "plot", "plot_mc2", "post_analysis"]
            for sendFolder in sendFolderSet:
                clientPath = os.path.join(projectPath, sessionID, sendFolder)
                state = os.system(f"sshpass -p md703 scp -r {clientPath} {host}:{sessionHostPathData}")
                if state != 0:
                    raise Exception(f"Sending error !! → {clientPath}")
            
            # send to SSD
            if ip_now != ip_host:
                sessionHostPathInfo = os.path.join(hostPathInfo, sessionID)
                sendFolderSet = ["json_output", "plot", "plot_mc2", "post_analysis"]
                for sendFolder in sendFolderSet:
                    clientPath = os.path.join(projectPath, sessionID, sendFolder)
                    state = os.system(f"sshpass -p md703 scp -r {clientPath} {host}:{sessionHostPathInfo}")
                    if state != 0:
                        raise Exception(f"Sending error !! → {clientPath}")
            
            # show time
            print(f"take {round(time.time()-startTime, 2)} s")