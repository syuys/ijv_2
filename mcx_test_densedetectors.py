# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 14:06:13 2021

@author: EricSyu
"""

from IPython import get_ipython
get_ipython().magic('clear')
get_ipython().magic('reset -f')
import postprocess
import matplotlib.pyplot as plt
plt.close("all")
import numpy as np
from scipy.interpolate import PchipInterpolator
from glob import glob
import os
import sys
import json
from random import randint

class MCX:
    
    # Initialize and call loadConfig()
    def __init__(self, projectName, sessionID):
        self.projectName = projectName
        self.sessionID = sessionID
        self.loadConfig(os.path.join(self.projectName, self.sessionID, "config.json"))
        self.createFolder()
        if not os.path.isfile(os.path.join(self.projectName, self.sessionID, "output", "json_output", "input_for_mcxpreview_hardware_closeup.json")):
            # make formal configuration of MCX
            self.makeMCXInput(self.wlSet[0])
            # make informal configuration of MCX for preview
            self.makeMCXInputForPreview()


    # Load configuration
    def loadConfig(self, configFile):
        # load
        with open(configFile) as f:
            self.config = json.load(f)
        
        # set PhotonBatch, but still not know why.
        if self.config["PhotonBatch"] > self.config["PhotonNum"]:
            self.config["PhotonBatch"] = self.config["PhotonNum"]
        
        # set simulated wavelength
        with open(self.config["SimulatedWavelengthPath"]) as f:
            self.wlSet = json.load(f)["Values"]
        self.wl = 745
        
        # load model_parameters
        with open(self.config["ModelParametersPath"]) as f:
            self.modelParameters = json.load(f)
        
        # load mcxInput setting template
        with open(self.config["MCXInputPath"]) as f:
            self.mcxInput = json.load(f)


    def createFolder(self):
        # set folder path        
        # main-path
        self.session = os.path.join(self.projectName, self.sessionID, "output")        
        # sub-path
        self.plot = os.path.join(self.session, "plot")
        self.plot_mc2 = os.path.join(self.session, "plot_mc2")
        self.post_analysis = os.path.join(self.session, "post_analysis")
        self.mcx_output = os.path.join(self.session, "mcx_output")
        self.json_output = os.path.join(self.session, "json_output")
        
        # create folder
        # main-path for this simulation session
        if not os.path.isdir(self.session):
            os.mkdir(self.session)
        # sub-path for saving plot
        if not os.path.isdir(self.plot):
            os.mkdir(self.plot)
        # sub-path for saving mc2 plot
        if not os.path.isdir(self.plot_mc2):
            os.mkdir(self.plot_mc2)
        # sub-path for saving post analysis (analysis after simulation if needed)
        if not os.path.isdir(self.post_analysis):
            os.mkdir(self.post_analysis)
        # sub-path for saving raw simulation output
        if not os.path.isdir(self.mcx_output):
            os.mkdir(self.mcx_output)
        # sub-path for saving MCX-used configuration
        if not os.path.isdir(self.json_output):
            os.mkdir(self.json_output) 
    

    # Main function to run simulation by passing formal configuration to MCX bin.
    def run(self):
        # main: run forward mcx
        if self.config["Type"] == "ijv":
            # loop each wavelength
            for wl in self.wlSet:
                # check if mcx input file is existed
                if not os.path.isfile(os.path.join(self.projectName, self.sessionID, "output", "json_output", "input_{}.json".format(wl))):
                    self.makeMCXInput(wl)                
                # get the current simulation progress and cv value.
                simulationResultPath = os.path.join(self.post_analysis, "{}_{}nm_simulation_result.json".format(self.config["SessionID"], wl))
                if os.path.isfile(simulationResultPath):  # if the simulation has run for a period of time            
                    with open(simulationResultPath) as f:
                        simulationResult = json.load(f)
                    existedOutputNum = simulationResult["RawSampleNum"]
                    reflectanceCV = simulationResult["GroupingSampleCV"].values()
                else:  # if there is no simulation result.json at all in post_analysis or just no simulation being run ever.
                    existedOutputNum = len(glob(os.path.join(self.mcx_output, "*{}nm*detp.jdat".format(wl))))
                    reflectanceCV = [1000]  # an arbitrarily large number                
                # start to run in a loop (if maximum of cv is not smaller than the cv threshold)
                while(max(reflectanceCV) > self.config["CVThreshold"]):                
                    # get the left number of repeat times to simulate
                    if self.config.get("SimulateWithSameSeed"):
                        needAddOutputNum = 1
                        # if we do simulation with same seed, there is no need to run MC more than one time.
                        if existedOutputNum >= 1:
                            break
                    else:
                        needAddOutputNum = self.config["RepeatTimes"] - existedOutputNum % self.config["RepeatTimes"]                    
                    # make command and run (based on existed number and need-add number)
                    for i in range(existedOutputNum, existedOutputNum+needAddOutputNum):
                        command = self.getCommand(wl, i)
                        sys.stdout.flush()
                        os.chdir(self.config["BinaryPath"])
                        print("Current position to run MCX:\n", os.getcwd(), end="\n\n")
                        print("Command sent to MCX:\n{}".format(command), end="\n\n")
                        print("∎∎ Start to run # {} ...".format(i), end=" ")
                        os.system(command)
                        print("Finished !! ∎∎", end="\n\n")
                        os.chdir("../..")
                        # update the RawSampleNum in simulation_result.json ("10" here is the cv base number)
                        # if i >= 10:
                        #     with open(simulationResultPath) as f:
                        #         simulationResult = json.load(f)
                        #     simulationResult["RawSampleNum"] = i + 1  # +1 is needed.
                        #     with open(simulationResultPath, "w") as f:
                        #         json.dump(simulationResult, f, indent=4)
                    # update the current existedOutputNum after the simulation loop
                    existedOutputNum = existedOutputNum + needAddOutputNum
                    # get the current state and print it.
                    raw, reflectance, reflectanceMean, reflectanceCV, totalPhoton, groupingNum = postprocess.analyzeReflectance(projectName, sessionID, wl)
                    print("Project name: {} \nSession name: {} \nReflectance mean: {} \nCV: {} \nNecessary photon num: {:.4e}".format(self.projectName, self.sessionID, reflectanceMean, reflectanceCV, totalPhoton*groupingNum), end="\n\n")
        else:
            raise Exception("'Type' in %s is invalid!\ntry 'ijv', 'artery' or 'phantom'." % self.config["SessionID"])


    # Create the user-defined command line flags for mcx
    def getCommand(self, wl, simOrder):        
        session_name = "{}_{}nm_{}".format(self.config["SessionID"], wl, simOrder)
        geometry_file = os.path.abspath(os.path.join(self.json_output, "input_{}.json".format(wl)))
        
        root = "\"%s\" " % os.path.join(os.path.abspath(self.session), "mcx_output")
        unitmm = "%f " % self.config["VoxelSize"]
        photon = "%d " % self.config["PhotonBatch"]
        num_batch = "%d " % (self.config["PhotonNum"]//self.config["PhotonBatch"])
        maxdetphoton = "10000000"
        # maxdetphoton = "%d" % (self.config["PhotonNum"]//5)
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
        command += "--gpu 1 "  # use first gpu
        command += "--autopilot 1 " 
        command += "--photon {} ".format(photon)
        command += "--repeat {} ".format(num_batch)
        command += "--normalize 1 "
        command += "--specular 1 "
        command += "--reflect 0 "
        if self.config.get("bcFlag"):
            command += "--bc {} ".format(self.config.get("bcFlag"))
        command += "--unitinmm {} ".format(unitmm)
        command += "--skipradius -2 " 
        command += "--array 0 " 
        command += "--dumpmask 0 "
        command += "--maxdetphoton {} ".format(maxdetphoton)
        command += "--srcfrom0 1 "
        command += "--savedetflag DPXVW "
        if self.config.get("DoSaveFlux"):
            command += "--outputtype {} ".format("E")
        else:
            command += "--outputtype {} ".format("X")
        command += "--outputformat {} ".format("jnii")
        if self.config.get("DoSaveTrajectory"):
            command += "--debug MP "
        else:
            command += "--debug P "
        # add output .mc2 20210428 above
        if self.config.get("replay"):
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
            if self.config.get("DoSaveFlux"):
                command += "--save2pt 1 "
            else:
                command += "--save2pt 0 "
            if self.config.get("SimulateWithSameSeed"):
                command += "--seed {} ".format(1)
            else:
                command += "--seed {} ".format(randint(0, 1000000000))

        return command


    def makeMCXInput(self, wl):
        # set Session
        self.mcxInput["Session"]["ID"] = self.config["SessionID"]
        
        # set Domain Media (optical properties of tissue model)
        self.setDomainMedia(wl)        
        
        # set Domain Dim
        self.mcxInput["Domain"]["Dim"] = [int(self.convertUnit(self.modelParameters["ModelSize"]["XSize"])),
                                          int(self.convertUnit(self.modelParameters["ModelSize"]["YSize"])),
                                          int(self.convertUnit(self.modelParameters["ModelSize"]["ZSize"]))
                                          ]
        
        # set Domain OriginType
        self.mcxInput["Domain"]["OriginType"] = 1
        
        # set Shapes
        self.setShapes()
        
        # set Optodes
        self.setOptodes()
        
        # save mcxInput to output/json_output
        with open(os.path.join(self.json_output, "input_{}.json".format(wl)), 'w') as f:
            json.dump(self.mcxInput, f, indent=4)
    
    
    def makeMCXInputForPreview(self):
        # Save mcxInput to output/json_output (these 2 files are for preview, not for simulation)
        # Need not set Session, Domain Media and Domain OriginType here. Just need to reset Shapes, Optodes for preview use.        
        
        # set Domain Dim for previewing whole model (Zoom out the dimension, ex: make the model only 80mm * 80mm * 25mm)
        fiberNum = len(self.modelParameters["HardwareParam"]["Detector"]["Fiber"])
        sds = self.modelParameters["HardwareParam"]["Detector"]["Fiber"][fiberNum//2]["SDS"]
        self.mcxInput["Domain"]["Dim"] = [int(self.convertUnit(sds*4)),
                                          int(self.convertUnit(sds*4)),
                                          int(self.convertUnit(self.modelParameters["GeoParam"]["CCADepth"]*2
                                                               + self.modelParameters["HardwareParam"]["Source"]["Holder"]["ZSize"]
                                                               )
                                              )
                                          ]        
        # set Shapes
        self.setShapes()        
        # set Optodes
        self.mcxInput["Optode"]["Detector"] = []  # initialize detector list fist
        self.setOptodes()        
        # change the source type because mcxpreview did not support "anglepattern"
        self.mcxInput["Optode"]["Source"]["Type"] = "pencil"
        # remove some source parameters because they are unnecessary when previewing
        del self.mcxInput["Optode"]["Source"]["Param1"]
        del self.mcxInput["Optode"]["Source"]["Param2"]
        del self.mcxInput["Optode"]["Source"]["Pattern"]        
        # save mcxInput to output/json_output ("closeup" in file name means we reduce the size of dimension for preview convenience)
        with open(os.path.join(self.json_output, "input_for_mcxpreview_wholemodel_closeup.json"), 'w') as f:
            json.dump(self.mcxInput, f, indent=4)
        
        # set Domain Dim for previewing hardware arrangement (Zoom out the dimension, ex: make the model only 80mm * 80mm * 25mm)
        self.mcxInput["Domain"]["Dim"][2] = int(self.convertUnit(self.modelParameters["HardwareParam"]["Source"]["Holder"]["ZSize"])) * 2
        # set Shapes and remove some unnecessary items
        self.setShapes()
        del self.mcxInput["Domain"]["Media"][6:]
        del self.mcxInput["Shapes"][6:]
        self.mcxInput["Shapes"][5]["Subgrid"]["Size"][2] = 5  # change z size of first tissue layer, which equals to z size of dimension - holder size [grid]        
        # save mcxInput to output/json_output ("closeup" in file name means we reduce the size of dimension for preview convenience)
        with open(os.path.join(self.json_output, "input_for_mcxpreview_hardware_closeup.json"), 'w') as f:
            json.dump(self.mcxInput, f, indent=4)


    def setDomainMedia(self, wl):
        # 0: Air
        self.mcxInput["Domain"]["Media"][0]["n"] = self.modelParameters["OptParam"]["Air"]["n"]
        self.mcxInput["Domain"]["Media"][0]["g"] = self.modelParameters["OptParam"]["Air"]["g"]
        self.mcxInput["Domain"]["Media"][0]["mua"] = 0
        self.mcxInput["Domain"]["Media"][0]["mus"] = self.modelParameters["OptParam"]["Air"]["mus"]
        # 1: Detector PLA
        self.mcxInput["Domain"]["Media"][1]["n"] = self.modelParameters["OptParam"]["PLA"]["n"]
        self.mcxInput["Domain"]["Media"][1]["g"] = self.modelParameters["OptParam"]["PLA"]["g"]
        self.mcxInput["Domain"]["Media"][1]["mua"] = 0
        self.mcxInput["Domain"]["Media"][1]["mus"] = self.modelParameters["OptParam"]["PLA"]["mus"]
        # 2: Detector Prism
        self.mcxInput["Domain"]["Media"][2]["n"] = self.modelParameters["OptParam"]["Prism"]["n"]
        self.mcxInput["Domain"]["Media"][2]["g"] = self.modelParameters["OptParam"]["Prism"]["g"]
        self.mcxInput["Domain"]["Media"][2]["mua"] = 0
        self.mcxInput["Domain"]["Media"][2]["mus"] = self.modelParameters["OptParam"]["Prism"]["mus"]
        # 3: Source PLA
        self.mcxInput["Domain"]["Media"][3]["n"] = self.modelParameters["OptParam"]["PLA"]["n"]
        self.mcxInput["Domain"]["Media"][3]["g"] = self.modelParameters["OptParam"]["PLA"]["g"]
        self.mcxInput["Domain"]["Media"][3]["mua"] = 0
        self.mcxInput["Domain"]["Media"][3]["mus"] = self.modelParameters["OptParam"]["PLA"]["mus"]        
        # 4: Source Air
        self.mcxInput["Domain"]["Media"][4]["n"] = self.modelParameters["OptParam"]["Air"]["n"]
        self.mcxInput["Domain"]["Media"][4]["g"] = self.modelParameters["OptParam"]["Air"]["g"]
        self.mcxInput["Domain"]["Media"][4]["mua"] = 0
        self.mcxInput["Domain"]["Media"][4]["mus"] = self.modelParameters["OptParam"]["Air"]["mus"]        
        # 5: Skin
        self.mcxInput["Domain"]["Media"][5]["n"] = self.modelParameters["OptParam"]["Skin"]["n"]
        self.mcxInput["Domain"]["Media"][5]["g"] = self.modelParameters["OptParam"]["Skin"]["g"]
        if self.config.get("DoSaveFlux"):
            self.mcxInput["Domain"]["Media"][5]["mua"] = 4e4
            self.mcxInput["Domain"]["Media"][5]["mus"] = 1e-4
        else:
            self.mcxInput["Domain"]["Media"][5]["mua"] = 0
            self.mcxInput["Domain"]["Media"][5]["mus"] = \
                self.calculateMus(wl, 
                                  musp745=self.modelParameters["OptParam"]["Skin"]["muspx"], 
                                  bmie=self.modelParameters["OptParam"]["Skin"]["bmie"], 
                                  g=self.modelParameters["OptParam"]["Skin"]["g"])
        # 6: Fat
        self.mcxInput["Domain"]["Media"][6]["n"] = self.modelParameters["OptParam"]["Fat"]["n"]
        self.mcxInput["Domain"]["Media"][6]["g"] = self.modelParameters["OptParam"]["Fat"]["g"]
        self.mcxInput["Domain"]["Media"][6]["mua"] = 0
        self.mcxInput["Domain"]["Media"][6]["mus"] = \
            self.calculateMus(wl, 
                              musp745=self.modelParameters["OptParam"]["Fat"]["muspx"], 
                              bmie=self.modelParameters["OptParam"]["Fat"]["bmie"], 
                              g=self.modelParameters["OptParam"]["Fat"]["g"])
        # 7: Muscle
        self.mcxInput["Domain"]["Media"][7]["n"] = self.modelParameters["OptParam"]["Muscle"]["n"]
        self.mcxInput["Domain"]["Media"][7]["g"] = self.modelParameters["OptParam"]["Muscle"]["g"]
        self.mcxInput["Domain"]["Media"][7]["mua"] = 0
        self.mcxInput["Domain"]["Media"][7]["mus"] = \
            self.calculateMus(wl, 
                              musp745=self.modelParameters["OptParam"]["Muscle"]["muspx"], 
                              bmie=self.modelParameters["OptParam"]["Muscle"]["bmie"], 
                              g=self.modelParameters["OptParam"]["Muscle"]["g"])
        # 8: IJV
        self.mcxInput["Domain"]["Media"][8]["n"] = self.modelParameters["OptParam"]["IJV"]["n"]
        self.mcxInput["Domain"]["Media"][8]["g"] = self.modelParameters["OptParam"]["IJV"]["g"]
        self.mcxInput["Domain"]["Media"][8]["mua"] = 0
        self.mcxInput["Domain"]["Media"][8]["mus"] = \
            self.calculateMus(wl, 
                              musp745=self.modelParameters["OptParam"]["IJV"]["muspx"], 
                              bmie=self.modelParameters["OptParam"]["IJV"]["bmie"], 
                              g=self.modelParameters["OptParam"]["IJV"]["g"])
        # 9: CCA
        self.mcxInput["Domain"]["Media"][9]["n"] = self.modelParameters["OptParam"]["CCA"]["n"]
        self.mcxInput["Domain"]["Media"][9]["g"] = self.modelParameters["OptParam"]["CCA"]["g"]
        self.mcxInput["Domain"]["Media"][9]["mua"] = 0
        self.mcxInput["Domain"]["Media"][9]["mus"] = \
            self.calculateMus(wl, 
                              musp745=self.modelParameters["OptParam"]["CCA"]["muspx"], 
                              bmie=self.modelParameters["OptParam"]["CCA"]["bmie"], 
                              g=self.modelParameters["OptParam"]["CCA"]["g"])

    
    def setShapes(self):
        # 0: Air
        self.mcxInput["Shapes"][0]["Grid"]["Size"] = self.mcxInput["Domain"]["Dim"] 
        
        # 1: Detector PLA (Here, help to extend detector holder for the simulation convenience)
        self.mcxInput["Shapes"][1]["Subgrid"]["Size"] = [int(self.convertUnit(self.modelParameters["HardwareParam"]["Detector"]["Holder"]["XSize"])),
                                                         int(self.convertUnit(2*self.modelParameters["HardwareParam"]["Detector"]["Holder"]["YSize"] 
                                                                              + self.modelParameters["HardwareParam"]["Source"]["Holder"]["YSize"])),
                                                         int(self.convertUnit(self.modelParameters["HardwareParam"]["Detector"]["Holder"]["ZSize"]))
                                                         ]
        self.mcxInput["Shapes"][1]["Subgrid"]["O"] = [int(self.mcxInput["Shapes"][0]["Grid"]["Size"][0]/2) 
                                                      - int(self.mcxInput["Shapes"][1]["Subgrid"]["Size"][0]/2),
                                                      int(self.mcxInput["Shapes"][0]["Grid"]["Size"][1]/2) 
                                                      - int(self.convertUnit(self.modelParameters["HardwareParam"]["Source"]["Holder"]["YSize"]/2))
                                                      - int(self.convertUnit(self.modelParameters["HardwareParam"]["Detector"]["Holder"]["YSize"])),
                                                      0
                                                      ]
        
        # 2: Detector Prism (Here, help to extend detector prism for the simulation convenience)
        self.mcxInput["Shapes"][2]["Subgrid"]["Size"] = [int(self.convertUnit(self.modelParameters["HardwareParam"]["Detector"]["Prism"]["XSize"])),
                                                         int(self.convertUnit(2*self.modelParameters["HardwareParam"]["Detector"]["Prism"]["YSize"] 
                                                                              + self.modelParameters["HardwareParam"]["Source"]["Holder"]["YSize"])),
                                                         int(self.convertUnit(self.modelParameters["HardwareParam"]["Detector"]["Prism"]["ZSize"]))
                                                         ]
        # self.mcxInput["Shapes"][4]["Subgrid"]["O"] = [int(self.mcxInput["Shapes"][0]["Grid"]["Size"][0]/2) 
        #                                               - int(self.mcxInput["Shapes"][4]["Subgrid"]["Size"][0]/2),
        #                                               int(self.mcxInput["Shapes"][0]["Grid"]["Size"][1]/2) 
        #                                               + int(self.mcxInput["Shapes"][1]["Subgrid"]["Size"][1]/2) 
        #                                               + int(self.mcxInput["Shapes"][2]["Subgrid"]["Size"][1]/2) 
        #                                               - int(self.mcxInput["Shapes"][4]["Subgrid"]["Size"][1]/2),
        #                                               0
        #                                               ]
        if len(self.modelParameters["HardwareParam"]["Detector"]["Fiber"]) <= 1:
            self.mcxInput["Shapes"][2]["Subgrid"]["O"] = [int(self.mcxInput["Shapes"][0]["Grid"]["Size"][0]/2)
                                                          - int(self.mcxInput["Shapes"][2]["Subgrid"]["Size"][0]/2),
                                                          int(self.mcxInput["Shapes"][0]["Grid"]["Size"][1]/2)
                                                          + int(self.convertUnit(self.modelParameters["HardwareParam"]["Detector"]["Fiber"][0]["SDS"]))
                                                          - int(self.mcxInput["Shapes"][2]["Subgrid"]["Size"][1]/2),
                                                          0
                                                          ]
        else:  # for simulating lots of sds at the same time
            self.mcxInput["Shapes"][2]["Subgrid"]["O"] = [int(self.mcxInput["Shapes"][0]["Grid"]["Size"][0]/2) 
                                                          - int(self.mcxInput["Shapes"][2]["Subgrid"]["Size"][0]/2),
                                                          int(self.mcxInput["Shapes"][0]["Grid"]["Size"][1]/2)
                                                          - int(self.convertUnit(self.modelParameters["HardwareParam"]["Source"]["Holder"]["YSize"]/2))
                                                          - int(self.convertUnit(self.modelParameters["HardwareParam"]["Detector"]["Prism"]["YSize"])),
                                                          0
                                                          ]
        
        # 3: Source PLA
        self.mcxInput["Shapes"][3]["Subgrid"]["Size"] = [int(self.convertUnit(self.modelParameters["HardwareParam"]["Source"]["Holder"]["XSize"])),
                                                         int(self.convertUnit(self.modelParameters["HardwareParam"]["Source"]["Holder"]["YSize"])),
                                                         int(self.convertUnit(self.modelParameters["HardwareParam"]["Source"]["Holder"]["ZSize"]))
                                                         ]
        self.mcxInput["Shapes"][3]["Subgrid"]["O"] = [int(self.mcxInput["Shapes"][0]["Grid"]["Size"][0]/2) 
                                                      - int(self.mcxInput["Shapes"][3]["Subgrid"]["Size"][0]/2),
                                                      int(self.mcxInput["Shapes"][0]["Grid"]["Size"][1]/2) 
                                                      - int(self.mcxInput["Shapes"][3]["Subgrid"]["Size"][1]/2),
                                                      0
                                                      ]
        
        # 4: Source Air
        self.mcxInput["Shapes"][4]["Cylinder"]["C0"] = [int(self.mcxInput["Shapes"][0]["Grid"]["Size"][0]/2),
                                                        int(self.mcxInput["Shapes"][0]["Grid"]["Size"][1]/2),
                                                        0
                                                        ]
        self.mcxInput["Shapes"][4]["Cylinder"]["C1"] = [int(self.mcxInput["Shapes"][0]["Grid"]["Size"][0]/2),
                                                        int(self.mcxInput["Shapes"][0]["Grid"]["Size"][1]/2),
                                                        self.mcxInput["Shapes"][3]["Subgrid"]["Size"][2]
                                                        ]
        self.mcxInput["Shapes"][4]["Cylinder"]["R"] = int(self.convertUnit(self.modelParameters["HardwareParam"]["Source"]["Holder"]["IrraWinRadius"]))
        
        # 5: Skin
        self.mcxInput["Shapes"][5]["Subgrid"]["Size"] = [self.mcxInput["Shapes"][0]["Grid"]["Size"][0],
                                                         self.mcxInput["Shapes"][0]["Grid"]["Size"][1],
                                                         int(self.convertUnit(self.modelParameters["GeoParam"]["SkinThk"]))
                                                         ]
        self.mcxInput["Shapes"][5]["Subgrid"]["O"] = [0,
                                                      0,
                                                      self.mcxInput["Shapes"][4]["Cylinder"]["C1"][2]
                                                      ]
        
        # 6: Fat
        self.mcxInput["Shapes"][6]["Subgrid"]["Size"] = [self.mcxInput["Shapes"][0]["Grid"]["Size"][0],
                                                         self.mcxInput["Shapes"][0]["Grid"]["Size"][1],
                                                         int(self.convertUnit(self.modelParameters["GeoParam"]["FatThk"]))
                                                         ]
        self.mcxInput["Shapes"][6]["Subgrid"]["O"] = [0,
                                                      0,
                                                      self.mcxInput["Shapes"][5]["Subgrid"]["O"][2] 
                                                      + self.mcxInput["Shapes"][5]["Subgrid"]["Size"][2]
                                                      ]
        
        # 7: Muscle (Reverse the setting order of "Size" and "Order".)
        self.mcxInput["Shapes"][7]["Subgrid"]["O"] = [0,
                                                      0,
                                                      self.mcxInput["Shapes"][6]["Subgrid"]["O"][2] 
                                                      + self.mcxInput["Shapes"][6]["Subgrid"]["Size"][2]
                                                      ]
        self.mcxInput["Shapes"][7]["Subgrid"]["Size"] = [self.mcxInput["Shapes"][0]["Grid"]["Size"][0],
                                                         self.mcxInput["Shapes"][0]["Grid"]["Size"][1],
                                                         self.mcxInput["Shapes"][0]["Grid"]["Size"][2] 
                                                         - self.mcxInput["Shapes"][7]["Subgrid"]["O"][2]
                                                         ]        
        
        # 8: IJV
        self.mcxInput["Shapes"][8]["Cylinder"]["C0"] = [int(self.mcxInput["Shapes"][0]["Grid"]["Size"][0]/2),
                                                        0,
                                                        self.convertUnit(self.modelParameters["GeoParam"]["IJVDepth"]) 
                                                        + self.mcxInput["Shapes"][5]["Subgrid"]["O"][2]
                                                        ]
        self.mcxInput["Shapes"][8]["Cylinder"]["C1"] = [int(self.mcxInput["Shapes"][0]["Grid"]["Size"][0]/2),
                                                        self.mcxInput["Shapes"][0]["Grid"]["Size"][1],
                                                        self.mcxInput["Shapes"][8]["Cylinder"]["C0"][2]
                                                        ]
        self.mcxInput["Shapes"][8]["Cylinder"]["R"] = self.convertUnit(self.modelParameters["GeoParam"]["IJVRadius"])
        
        # 9: CCA
        ccax = self.mcxInput["Shapes"][8]["Cylinder"]["C0"][0] - self.convertUnit(np.sqrt(self.modelParameters["GeoParam"]["IJVCCADist"]**2 
                                                                                          - (self.modelParameters["GeoParam"]["CCADepth"] - self.modelParameters["GeoParam"]["IJVDepth"])**2
                                                                                          )
                                                                                  )
        self.mcxInput["Shapes"][9]["Cylinder"]["C0"] = [ccax,
                                                        0,
                                                        self.convertUnit(self.modelParameters["GeoParam"]["CCADepth"]) 
                                                        + self.mcxInput["Shapes"][5]["Subgrid"]["O"][2]
                                                        ]
        self.mcxInput["Shapes"][9]["Cylinder"]["C1"] = [ccax,
                                                        self.mcxInput["Shapes"][0]["Grid"]["Size"][1],
                                                        self.mcxInput["Shapes"][9]["Cylinder"]["C0"][2]
                                                        ]
        self.mcxInput["Shapes"][9]["Cylinder"]["R"] = self.convertUnit(self.modelParameters["GeoParam"]["CCARadius"])


    def setOptodes(self):
        # detector (help to extend to left and right)
        for fiber in self.modelParameters["HardwareParam"]["Detector"]["Fiber"]:
            # left - top
            self.mcxInput["Optode"]["Detector"].append({"R": self.convertUnit(fiber["Radius"]),
                                                        "Pos": [self.mcxInput["Shapes"][0]["Grid"]["Size"][0]/2 - 2*self.convertUnit(fiber["Radius"]),
                                                                self.mcxInput["Shapes"][0]["Grid"]["Size"][1]/2 
                                                                + self.convertUnit(fiber["SDS"]),
                                                                0
                                                                ]
                                                        })
            # left - bottom
            self.mcxInput["Optode"]["Detector"].append({"R": self.convertUnit(fiber["Radius"]),
                                                        "Pos": [self.mcxInput["Shapes"][0]["Grid"]["Size"][0]/2 - 2*self.convertUnit(fiber["Radius"]),
                                                                self.mcxInput["Shapes"][0]["Grid"]["Size"][1]/2 
                                                                - self.convertUnit(fiber["SDS"]),
                                                                0
                                                                ]
                                                        })
            # middle - top (original)
            self.mcxInput["Optode"]["Detector"].append({"R": self.convertUnit(fiber["Radius"]),
                                                        "Pos": [self.mcxInput["Shapes"][0]["Grid"]["Size"][0]/2,
                                                                self.mcxInput["Shapes"][0]["Grid"]["Size"][1]/2 
                                                                + self.convertUnit(fiber["SDS"]),
                                                                0
                                                                ]
                                                        })
            # middle - bottom
            self.mcxInput["Optode"]["Detector"].append({"R": self.convertUnit(fiber["Radius"]),
                                                        "Pos": [self.mcxInput["Shapes"][0]["Grid"]["Size"][0]/2,
                                                                self.mcxInput["Shapes"][0]["Grid"]["Size"][1]/2 
                                                                - self.convertUnit(fiber["SDS"]),
                                                                0
                                                                ]
                                                        })
            # right - top
            self.mcxInput["Optode"]["Detector"].append({"R": self.convertUnit(fiber["Radius"]),
                                                        "Pos": [self.mcxInput["Shapes"][0]["Grid"]["Size"][0]/2 + 2*self.convertUnit(fiber["Radius"]),
                                                                self.mcxInput["Shapes"][0]["Grid"]["Size"][1]/2 
                                                                + self.convertUnit(fiber["SDS"]),
                                                                0
                                                                ]
                                                        })
            # right - bottom
            self.mcxInput["Optode"]["Detector"].append({"R": self.convertUnit(fiber["Radius"]),
                                                        "Pos": [self.mcxInput["Shapes"][0]["Grid"]["Size"][0]/2 + 2*self.convertUnit(fiber["Radius"]),
                                                                self.mcxInput["Shapes"][0]["Grid"]["Size"][1]/2 
                                                                - self.convertUnit(fiber["SDS"]),
                                                                0
                                                                ]
                                                        })
        # self.mcxInput["Optode"]["Detector"][0]["R"] = self.convertUnit(self.modelParameters["HardwareParam"]["Detector"]["Fiber1"]["Radius"])
        # self.mcxInput["Optode"]["Detector"][0]["Pos"] = [self.mcxInput["Shapes"][0]["Grid"]["Size"][0]/2,
        #                                                  self.mcxInput["Shapes"][0]["Grid"]["Size"][1]/2 
        #                                                  + self.convertUnit(self.modelParameters["HardwareParam"]["Detector"]["Fiber1"]["SDS"]),
        #                                                  0
        #                                                  ]
        
        # source
        # sampling angles based on the radiation pattern distribution
        ledProfileIn3D = np.genfromtxt(self.modelParameters["HardwareParam"]["Source"]["Beam"]["ProfilePath"], delimiter=",")
        angle = ledProfileIn3D[:, 0]
        cdf = np.cumsum(ledProfileIn3D[:, 1])
        inversecdf = PchipInterpolator(cdf, angle)        
        samplingSeeds = np.linspace(0, 1, num=int(self.modelParameters["HardwareParam"]["Source"]["LED"]["SamplingNumOfRadiationPattern"]))
        samplingAngles = inversecdf(samplingSeeds)
        # set source position
        self.mcxInput["Optode"]["Source"]["Pos"] = [self.mcxInput["Shapes"][0]["Grid"]["Size"][0]/2,
                                                    self.mcxInput["Shapes"][0]["Grid"]["Size"][1]/2,
                                                    self.mcxInput["Shapes"][5]["Subgrid"]["O"][2]-1e-6  # need to minus a very small value here. (if not, irradiating distribution will disappear)
                                                    ]        
        # set source type
        self.mcxInput["Optode"]["Source"]["Type"] = self.modelParameters["HardwareParam"]["Source"]["Beam"]["Type"]
        # set additional parameters about led arrangement (please refer to the hackMD-ijvNotebooks records on 2021/04/09)
        self.mcxInput["Optode"]["Source"]["Param1"] = [int(self.modelParameters["HardwareParam"]["Source"]["LED"]["SamplingNumOfRadiationPattern"]), 
                                                       self.convertUnit(self.modelParameters["HardwareParam"]["Source"]["LED"]["XSize"]), 
                                                       self.convertUnit(self.modelParameters["HardwareParam"]["Source"]["LED"]["YSize"]), 
                                                       0
                                                       ]
        self.mcxInput["Optode"]["Source"]["Param2"] = [0, 
                                                       0, 
                                                       self.convertUnit(self.modelParameters["HardwareParam"]["Source"]["Holder"]["IrraWinRadius"]), 
                                                       self.convertUnit(self.modelParameters["HardwareParam"]["Source"]["LED"]["Surf2Win"])
                                                       ]
        # set formal pattern
        self.mcxInput["Optode"]["Source"]["Pattern"] = {}
        self.mcxInput["Optode"]["Source"]["Pattern"]["Nx"] = int(self.modelParameters["HardwareParam"]["Source"]["LED"]["SamplingNumOfRadiationPattern"])
        self.mcxInput["Optode"]["Source"]["Pattern"]["Ny"] = 1
        self.mcxInput["Optode"]["Source"]["Pattern"]["Nz"] = 1
        self.mcxInput["Optode"]["Source"]["Pattern"]["Data"] = np.deg2rad(samplingAngles).tolist()
    

    def calculateMus(self, wl, musp745, bmie, g):        
        musp = musp745 * (wl/745) ** (-bmie)
        mus = musp/(1-g) * 0.1  # *0.1 means unit convertion from 1/cm to 1/mm
        return mus
    
    
    def convertUnit(self, length):
        """
        Do unit conversion.

        Parameters
        ----------
        length : int or float
            The unit of length is [mm].

        Returns
        -------
        numGrid : int or float
            Number of grid, for MCX simulation.

        """
        numGrid = length / self.config["VoxelSize"]
        return numGrid

# %% Run
if __name__ == "__main__":
    # # parameters
    # projectName = "20210810_prism_effect_test"
    # sessionID = "extended_prism"
    # cvThold = 0.02
    
    # # calculate reflectance first
    # raw, reflectance, reflectanceMean, reflectanceCV, totalPhoton, groupingNum = postprocess.analyzeReflectance(projectName, sessionID)
    # print("Session name: {} \nReflectance mean: {} \nCV: {} \nNecessary photon num: {:.2e}".format(sessionID, reflectanceMean, reflectanceCV, totalPhoton*groupingNum), end="\n\n")
    
    # # initialize
    # simulator = MCX(projectName, sessionID)
    
    # # run
    # while(max(reflectanceCV) > cvThold):
    #     # run forward mcx
    #     simulator.run()
    #     # check cv and print info
    #     raw, reflectance, reflectanceMean, reflectanceCV, totalPhoton, groupingNum = postprocess.analyzeReflectance(projectName, sessionID)
    #     print("Session name: {} \nReflectance mean: {} \nCV: {} \nNecessary photon num: {:.2e}".format(sessionID, reflectanceMean, reflectanceCV, totalPhoton*groupingNum), end="\n\n")
    
    # parameters
    projectName = "20211026_newmodel_size_test"
    sessionID = "test"
    
    # initialize
    simulator = MCX(projectName, sessionID)
    
    # run forward mcx
    simulator.run()



