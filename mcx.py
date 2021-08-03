# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 14:06:13 2021

@author: EricSyu
"""

import numpy as np
from scipy.interpolate import PchipInterpolator
import os
import sys
import json
from random import randint

class MCX:
    
    # Initialize and call loadConfig()
    def __init__(self, configFile):
        self.loadConfig(configFile)


    # Load configuration
    def loadConfig(self, configFile):
        # load
        with open(configFile) as f:
            self.config = json.load(f)
        
        # set photon_batch, but still not know why.
        if self.config["photon_batch"] > self.config["num_photon"]:
            self.config["photon_batch"] = self.config["num_photon"]
        
        # mcx_input setting
        with open(self.config["mcx_input"]) as f:
            self.mcx_input = json.load(f)
        
        # folder path setting        
        # main-path
        self.session = os.path.join("output", self.config["session_id"])        
        # sub-path
        self.plot = os.path.join(self.session, "plot")
        self.plot_mc2 = os.path.join(self.session, "plot_mc2")
        self.research_analysis = os.path.join(self.session, "research_analysis")
        self.mcx_output = os.path.join(self.session, "mcx_output")
        self.json_output = os.path.join(self.session, "json_output")


    # Main function to run simulation by passing formal configuration to MCX bin.
    def run(self):
        # main-path for this simulation session
        if not os.path.isdir(self.session):
            os.mkdir(self.session)
        # sub-path for saving plot
        if not os.path.isdir(self.plot):
            os.mkdir(self.plot)
        # sub-path for saving mc2 plot
        if not os.path.isdir(self.plot_mc2):
            os.mkdir(self.plot_mc2)
        # sub-path for saving research analysis (analysis after simulation if needed)
        if not os.path.isdir(self.research_analysis):
            os.mkdir(self.research_analysis)
        # sub-path for saving raw simulation output
        if not os.path.isdir(self.mcx_output):
            os.mkdir(self.mcx_output)
        # sub-path for saving MCX-used configuration
        if not os.path.isdir(self.json_output):
            os.mkdir(self.json_output)            

        # main: run forward mcx
        if self.config["type"] == "ijv":
            # make formal configuration of MCX
            self.makeInput()
            # make command and run (repeat {user-specified repeatTimes} times)
            for i in range(self.config["repeatTimes"]):
                command = self.getCommand(i)
                sys.stdout.flush()
                os.chdir(self.config["binary_path"])
                print("Current position to run MCX:\n", os.getcwd(), end="\n\n")
                print("Command sent to MCX:\n{}".format(command), end="\n\n")
                print("Start to run # {}".format(i), end="\n\n")
                os.system(command)
                os.chdir("../..")
        else:
            raise Exception("'type' in %s is invalid!\ntry 'ijv', 'artery' or 'phantom'." % self.config["session_id"])


    # Create the user-defined command line flags for mcx
    def getCommand(self, simOrder):        
        session_name = "{}_{}".format(self.config["session_id"], simOrder)
        geometry_file = os.path.abspath(os.path.join(self.json_output, "input.json"))
        
        root = "\"%s\" " % os.path.join(os.path.abspath(self.session), "mcx_output")
        unitmm = "%f " % self.config["voxel_size"]
        photon = "%d " % self.config["photon_batch"]
        num_batch = "%d " % (self.config["num_photon"]//self.config["photon_batch"])
        maxdetphoton = "10000000"
        # maxdetphoton = "%d" % (self.config["num_photon"]//5)
        # save_mc2 = "0 " if self.config["train"] else "1 "
        # mc2 is seldom used

        if os.name == "posix":
            # linux
            command = "./mcx"
        elif os.name == "nt":
            # windows
            command = "mcx.exe"
        else:
            command = "./mcx"
        command += " --session {} ".format(session_name)
        command += "--input {} ".format(geometry_file)
        command += "--root {} ".format(root)
        command += "--gpu 1 " 
        command += "--autopilot 1 " 
        command += "--photon {} ".format(photon)
        command += "--repeat {} ".format(num_batch)
        command += "--normalize 1 " 
        command += "--reflect 0 "
        command += "--unitinmm {} ".format(unitmm)
        command += "--skipradius -2 " 
        command += "--array 0 " 
        command += "--dumpmask 0 " 
        command += "--maxdetphoton {} ".format(maxdetphoton)
        command += "--srcfrom0 1 "
        command += "--savedetflag DPXVW "
        command += "--outputtype {} ".format("X")
        command += "--outputformat {} ".format("jnii")
        command += "--debug P "
        # add output .mc2 20210428 above
        if self.config.get("replay", None):
            command += "--saveseed 1 "
            if self.config.get("replay_mch", None):
                command += "--save2pt 1 "
                command += "--replaydet 1 "
                command += "--seed {} ".format(self.config["replay_mch"])
                command += "--outputtype {} ".format("J")
            else:
                command += "--seed {} ".format(randint(0, 1000000000))
                command += "--save2pt 0 "
        else:
            command += "--saveseed 1 "
            command += "--save2pt 0 "
            command += "--seed {} ".format(randint(0, 1000000000))
            # command += "--seed {} ".format(1)

        return command
    
    def makeInput(self):
        # set session
        self.mcx_input["Session"]["ID"] = self.config["session_id"]
        self.mcx_input["Session"]["Photons"] = self.config["num_photon"]
        
        # set optical properties of tissue model
        wl = 745
        # 0: Air
        self.mcx_input["Domain"]["Media"][0]["n"] = 1
        self.mcx_input["Domain"]["Media"][0]["g"] = 1
        self.mcx_input["Domain"]["Media"][0]["mua"] = 0
        self.mcx_input["Domain"]["Media"][0]["mus"] = 0
        # 1: Source PLA
        self.mcx_input["Domain"]["Media"][1]["n"] = 1.45
        self.mcx_input["Domain"]["Media"][1]["g"] = 1
        self.mcx_input["Domain"]["Media"][1]["mua"] = 0
        self.mcx_input["Domain"]["Media"][1]["mus"] = 1e-4
        # 2: Detector PLA
        self.mcx_input["Domain"]["Media"][2]["n"] = 1.45
        self.mcx_input["Domain"]["Media"][2]["g"] = 1
        self.mcx_input["Domain"]["Media"][2]["mua"] = 0
        self.mcx_input["Domain"]["Media"][2]["mus"] = 1e-4
        # 3: Source Air
        self.mcx_input["Domain"]["Media"][3]["n"] = 1
        self.mcx_input["Domain"]["Media"][3]["g"] = 1
        self.mcx_input["Domain"]["Media"][3]["mua"] = 0
        self.mcx_input["Domain"]["Media"][3]["mus"] = 0
        # 4: Detector Prism
        self.mcx_input["Domain"]["Media"][4]["n"] = 1.51
        self.mcx_input["Domain"]["Media"][4]["g"] = 1
        self.mcx_input["Domain"]["Media"][4]["mua"] = 0
        self.mcx_input["Domain"]["Media"][4]["mus"] = 1e-4
        # 5: Skin
        self.mcx_input["Domain"]["Media"][5]["n"] = 1.42
        self.mcx_input["Domain"]["Media"][5]["g"] = 0.9
        self.mcx_input["Domain"]["Media"][5]["mua"] = 0
        self.mcx_input["Domain"]["Media"][5]["mus"] = \
            self.calculateMus(wl, musp745=24, bmie=1.6, 
                              g=self.mcx_input["Domain"]["Media"][5]["g"])
        # 6: Fat
        self.mcx_input["Domain"]["Media"][6]["n"] = 1.4
        self.mcx_input["Domain"]["Media"][6]["g"] = 0.9
        self.mcx_input["Domain"]["Media"][6]["mua"] = 0
        self.mcx_input["Domain"]["Media"][6]["mus"] = \
            self.calculateMus(wl, musp745=17, bmie=0.7, 
                              g=self.mcx_input["Domain"]["Media"][6]["g"])
        # 7: Muscle
        self.mcx_input["Domain"]["Media"][7]["n"] = 1.4
        self.mcx_input["Domain"]["Media"][7]["g"] = 0.9
        self.mcx_input["Domain"]["Media"][7]["mua"] = 0
        self.mcx_input["Domain"]["Media"][7]["mus"] = \
            self.calculateMus(wl, musp745=6, bmie=1.9, 
                              g=self.mcx_input["Domain"]["Media"][7]["g"])
        # 8: IJV
        self.mcx_input["Domain"]["Media"][8]["n"] = 1.4
        self.mcx_input["Domain"]["Media"][8]["g"] = 0.99
        self.mcx_input["Domain"]["Media"][8]["mua"] = 0
        self.mcx_input["Domain"]["Media"][8]["mus"] = 100
        # 9: CCA
        self.mcx_input["Domain"]["Media"][9]["n"] = 1.4
        self.mcx_input["Domain"]["Media"][9]["g"] = 0.99
        self.mcx_input["Domain"]["Media"][9]["mua"] = 0
        self.mcx_input["Domain"]["Media"][9]["mus"] = 100
        
        # set light source of tissue model
        # sampling angles based on the radiation pattern distribution
        sampling_num = 100000
        
        LED_profile_in3D = np.genfromtxt(self.config["sourcePattern_path"], delimiter=",")
        angle = LED_profile_in3D[:, 0]
        cdf = np.cumsum(LED_profile_in3D[:, 1])
        inverse_cdf = PchipInterpolator(cdf, angle)
        
        samplingSeeds = np.linspace(0, 1, num=sampling_num)
        samplingAngles = inverse_cdf(samplingSeeds)
        
        # write the sampling angle array into mcx_input
        self.mcx_input["Optode"]["Source"]["Type"] = "anglepattern"
        self.mcx_input["Optode"]["Source"]["Pos"] = [self.mcx_input["Domain"]["Dim"][0] / 2,
                                                     self.mcx_input["Domain"]["Dim"][1] / 2, 
                                                     20-0.00001]
        
        # Set the source-related parameters !!!
        self.mcx_input["Optode"]["Source"]["Param1"] = [sampling_num, 10.2, 7.2, 0]
        self.mcx_input["Optode"]["Source"]["Param2"] = [self.mcx_input["Domain"]["Dim"][0], 
                                                        self.mcx_input["Domain"]["Dim"][1], 
                                                        0, 
                                                        24]
        self.mcx_input["Optode"]["Source"]["Pattern"] = {}
        self.mcx_input["Optode"]["Source"]["Pattern"]["Nx"] = sampling_num
        self.mcx_input["Optode"]["Source"]["Pattern"]["Ny"] = 1
        self.mcx_input["Optode"]["Source"]["Pattern"]["Nz"] = 1
        self.mcx_input["Optode"]["Source"]["Pattern"]["Data"] = np.deg2rad(samplingAngles).tolist()        
        
        with open(os.path.join(self.json_output, "input.json"), 'w') as f:
            json.dump(self.mcx_input, f, indent=4)


    def calculateMus(self, wl, musp745, bmie, g):        
        musp = musp745 * (wl/745) ** (-bmie)
        mus = musp/(1-g) * 0.1
        return mus


if __name__ == "__main__":
    import jdata as jd
    import matplotlib.pyplot as plt
    plt.rcParams.update({"mathtext.default": "regular"})
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["figure.dpi"] = 300
    
    # config file place
    config = "configs/single_detector.json"
    
    # start
    mcx = MCX(config)
    
    # run forward mcx
    mcx.run()
    
    #%% plot energy density matrix
    pt = jd.load('output/ijv/mcx_output/ijv.jnii')
    densityData = pt["NIFTIData"]
    gridStep = 5
    mcx_gridSize = mcx.config["voxel_size"]    
    
    surfaceDensity = densityData[:, :, 0]    
    norm = surfaceDensity / surfaceDensity.max()
    plt.imshow(norm.T, cmap="jet") # Normalization !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 20210527
    cbar = plt.colorbar()
    cbar.set_label("Energy density [J/mm^3]", rotation=-90, labelpad=15)
    plt.yticks(np.arange(-0.5, surfaceDensity.shape[1]-0.5+0.001, step=gridStep), 
               np.arange(0, surfaceDensity.shape[1]*mcx_gridSize+0.001, step=gridStep*mcx_gridSize), 
               fontsize="x-small")
    plt.ylabel("Y [mm]")
    plt.xticks(np.arange(-0.5, surfaceDensity.shape[0]-0.5+0.001, step=gridStep), 
               np.arange(0, surfaceDensity.shape[0]*mcx_gridSize+0.001, step=gridStep*mcx_gridSize),
               rotation=-90, 
               fontsize="x-small")
    plt.xlabel("X [mm]")
    plt.grid(color='w', linestyle='-', linewidth=0.5)
    plt.title("Surface density _ Outer (first layer of grids)")
    plt.show()