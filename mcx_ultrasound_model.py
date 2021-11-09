# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 14:06:13 2021

@author: EricSyu
"""

# from IPython import get_ipython
# get_ipython().magic('clear')
# get_ipython().magic('reset -f')
import postprocess
import matplotlib.pyplot as plt
plt.close("all")
import numpy as np
from scipy.interpolate import PchipInterpolator
import scipy.io as sio
import cv2
import jdata as jd
from glob import glob
import os
import sys
import json

class MCX:
    
    # Initialize and call loadConfig()
    def __init__(self, sessionID):
        self.sessionID = sessionID
        self.loadConfig(os.path.join(self.sessionID, "config.json"))
        self.createFolder()
        # if not os.path.isfile(os.path.join(self.projectName, self.sessionID, "output", "json_output", "input_for_mcxpreview_hardware_closeup.json")):
        #     # make formal configuration of MCX
        #     self.makeMCXInput(self.wlSet[0])
        #     # make informal configuration of MCX for preview
        #     self.makeMCXInputForPreview()


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
        currentDir = os.getcwd()
        os.chdir("../")
        with open(self.config["MCXInputPath"]) as f:
            self.mcxInput = json.load(f)
        os.chdir(currentDir)


    def createFolder(self):
        # set folder path        
        # main-path
        self.session = os.path.join(self.sessionID, "output")        
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
                if not os.path.isfile(os.path.join(self.json_output, "input_{}.json".format(wl))):
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
                    if "--save2pt 1" in str(self.config["CustomizedCommands"]) or "--seed 1" in str(self.config["CustomizedCommands"]):
                        needAddOutputNum = 1
                        if existedOutputNum >= 1:
                            break
                    else:
                        needAddOutputNum = self.config["RepeatTimes"] - existedOutputNum % self.config["RepeatTimes"]                    
                    # make command and run (based on existed number and need-add number)
                    for i in range(existedOutputNum, existedOutputNum+needAddOutputNum):
                        for customizedCommand in self.config["CustomizedCommands"]:
                            command = self.getCommand(wl, i, customizedCommand)                            
                            # run
                            sys.stdout.flush()
                            currentDir = os.getcwd()
                            os.chdir("../")
                            os.chdir(self.config["BinaryPath"])
                            print("Current position to run MCX:\n", os.getcwd(), end="\n\n")
                            print("Command sent to MCX:\n{}".format(command), end="\n\n")
                            print("∎∎ Start to run # {} ...".format(i), end=" ")
                            os.system(command)
                            print("Finished !! ∎∎", end="\n\n")
                            os.chdir(currentDir)
                        # update sample num of simulation_result.json
                        if i >= 10:
                            with open(simulationResultPath) as f:
                                simulationResult = json.load(f)
                            simulationResult["RawSampleNum"] = i + 1  # +1 is needed.
                            with open(simulationResultPath, "w") as f:
                                json.dump(simulationResult, f, indent=4)
                    # remove .mc2
                    mc2OutputPathSet = glob(os.path.abspath(os.path.join(self.mcx_output, "*.mc2")))
                    for mc2OutputPath in mc2OutputPathSet:
                        os.remove(mc2OutputPath)
                    # update the current existedOutputNum after the simulation loop
                    existedOutputNum = existedOutputNum + needAddOutputNum
                    # get the current state and print it.
                    raw, reflectance, reflectanceMean, reflectanceCV, totalPhoton, groupingNum = postprocess.analyzeReflectance(self.sessionID, wl)
                    print("Session name: {} \nReflectance mean: {} \nCV: {} \nNecessary photon num: {:.4e}".format(self.sessionID, reflectanceMean, reflectanceCV, totalPhoton*groupingNum), end="\n\n")
        else:
            raise Exception("'Type' in %s is invalid!\ntry 'ijv', 'artery' or 'phantom'." % self.config["SessionID"])


    def replay(self, volDim):
        detOutputPathSet = glob(os.path.join(self.mcx_output, "*.mch"))  # about paths of detected photon data
        detOutputPathSet.sort(key=lambda x: int(x.split("_")[-1].replace(".mch", "")))
        for wl in self.wlSet:
            self.makeMCXInput(wl, designatedVolDim=volDim)
            customizedCommand = self.config["CustomizedCommands"][1]
            for i in range(len(detOutputPathSet)):
                command = self.getCommand(wl, i, customizedCommand)
                # run
                sys.stdout.flush()
                currentDir = os.getcwd()
                os.chdir("../")
                os.chdir(self.config["BinaryPath"])
                print("Current position to run MCX:\n", os.getcwd(), end="\n\n")
                print("Command sent to MCX:\n{}".format(command), end="\n\n")
                print("∎∎ Start to run # {} ...".format(i), end=" ")
                os.system(command)
                print("Finished !! ∎∎", end="\n\n")
                os.chdir(currentDir)
            # remove .jnii
            jniiOutputPathSet = glob(os.path.abspath(os.path.join(self.mcx_output, "*.jnii")))
            for jniiOutputPath in jniiOutputPathSet:
                os.remove(jniiOutputPath)
                

    
    # Create the user-defined command line flags for mcx
    def getCommand(self, wl, simOrder, customizedCommand): 
        # basic setting
        sessionName = "{}_{}nm_{}".format(self.config["SessionID"], wl, simOrder)
        geometryFile = os.path.abspath(os.path.join(self.json_output, "input_{}.json".format(wl)))
        root = os.path.join(os.path.abspath(self.session), "mcx_output")
        numBatch = int(self.config["PhotonNum"]//self.config["PhotonBatch"])
        
        # make up command
        if os.name == "posix":
            # linux
            command = "./mcx"
        elif os.name == "nt":
            # windows
            command = "mcx.exe"
        else:
            command = "./mcx"
        command += " --session {} ".format(sessionName)
        command += "--input {} ".format(geometryFile)
        command += "--root {} ".format(root)
        command += "--gpu 1 "  # use first gpu
        command += "--autopilot 1 " 
        command += "--photon {} ".format(self.config["PhotonBatch"])
        command += "--repeat {} ".format(numBatch)
        command += "--normalize 1 "
        command += "--bc aaaaaa "
        command += "--unitinmm {} ".format(self.config["VoxelSize"])
        command += "--skipradius -2 " 
        command += "--array 0 " 
        command += "--dumpmask 0 "
        command += "--maxdetphoton {} ".format(1e7)
        command += "--srcfrom0 1 "
        command += "--saveseed 1 "
        command += "--debug P "
        if "mch" in customizedCommand:  # for replay
            customizedCommand = customizedCommand.replace("mch", os.path.abspath(os.path.join(self.mcx_output, "{}.mch".format(sessionName))))
        command += customizedCommand

        return command


    def makeMCXInput(self, wl, designatedVolDim=None):
        """

        Parameters
        ----------
        wl : TYPE
            DESCRIPTION.
        designatedVolDim : TYPE, optional
            If designatedVolDim is given, its form should be [XSize, YSize, ZSize] in mm.

        Returns
        -------
        None.

        """
        # set Session
        self.mcxInput["Session"]["ID"] = self.config["SessionID"]
        
        # set Domain Media (optical properties of tissue model)
        self.setDomainMedia(wl)        
        
        # set Domain Dim
        if designatedVolDim:
            self.mcxInput["Domain"]["Dim"] = [int(self.convertUnit(designatedVolDim[0])),
                                              int(self.convertUnit(designatedVolDim[1])),
                                              int(self.convertUnit(designatedVolDim[2]))
                                              ]
        else:
            self.mcxInput["Domain"]["Dim"] = [int(self.convertUnit(self.modelParameters["ModelSize"]["XSize"])),
                                              int(self.convertUnit(self.modelParameters["ModelSize"]["YSize"])),
                                              int(self.convertUnit(self.modelParameters["ModelSize"]["ZSize"]))
                                              ]
        
        # set Domain OriginType
        self.mcxInput["Domain"]["OriginType"] = 1
        
        # set Shapes
        # self.setShapes()
        self.setShapesFromUltrasound()
        
        # set Optodes
        self.setOptodes()
        
        # save mcxInput to output/json_output
        jd.save(jd.encode(self.mcxInput, {'compression':'zlib','base64':1}), 
                os.path.join(self.json_output, "input_{}.json".format(wl)))
        self.mcxInput["Optode"]["Source"] = {"Type": "pencil", 
                                             "Pos": self.mcxInput["Optode"]["Source"]["Pos"],
                                             "Dir": [0.0, 0.0, 1.0]}
        jd.save(jd.encode(self.mcxInput, {'compression':'zlib','base64':1}), 
                os.path.join(self.json_output, "input_{}_forpreview.json".format(wl)))
        # with open(os.path.join(self.json_output, "input_{}.json".format(wl)), 'w') as f:
        #     json.dump(self.mcxInput, f, indent=4)
    
    
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
        # 0: Fiber
        self.mcxInput["Domain"]["Media"][0]["n"] = self.modelParameters["OptParam"]["Fiber"]["n"]
        self.mcxInput["Domain"]["Media"][0]["g"] = self.modelParameters["OptParam"]["Fiber"]["g"]
        self.mcxInput["Domain"]["Media"][0]["mua"] = 0
        self.mcxInput["Domain"]["Media"][0]["mus"] = self.modelParameters["OptParam"]["Fiber"]["mus"]
        # 1: Air
        self.mcxInput["Domain"]["Media"][1]["n"] = self.modelParameters["OptParam"]["Air"]["n"]
        self.mcxInput["Domain"]["Media"][1]["g"] = self.modelParameters["OptParam"]["Air"]["g"]
        self.mcxInput["Domain"]["Media"][1]["mua"] = 0
        self.mcxInput["Domain"]["Media"][1]["mus"] = self.modelParameters["OptParam"]["Air"]["mus"]
        # 2: PLA
        self.mcxInput["Domain"]["Media"][2]["n"] = self.modelParameters["OptParam"]["PLA"]["n"]
        self.mcxInput["Domain"]["Media"][2]["g"] = self.modelParameters["OptParam"]["PLA"]["g"]
        self.mcxInput["Domain"]["Media"][2]["mua"] = 0
        self.mcxInput["Domain"]["Media"][2]["mus"] = self.modelParameters["OptParam"]["PLA"]["mus"]
        # 3:  Prism
        self.mcxInput["Domain"]["Media"][3]["n"] = self.modelParameters["OptParam"]["Prism"]["n"]
        self.mcxInput["Domain"]["Media"][3]["g"] = self.modelParameters["OptParam"]["Prism"]["g"]
        self.mcxInput["Domain"]["Media"][3]["mua"] = 0
        self.mcxInput["Domain"]["Media"][3]["mus"] = self.modelParameters["OptParam"]["Prism"]["mus"]
        # 4: Skin
        self.mcxInput["Domain"]["Media"][4]["n"] = self.modelParameters["OptParam"]["Skin"]["n"]
        self.mcxInput["Domain"]["Media"][4]["g"] = self.modelParameters["OptParam"]["Skin"]["g"]
        if "--save2pt 1" in str(self.config["CustomizedCommands"]):
            self.mcxInput["Domain"]["Media"][4]["mua"] = 4e4
            self.mcxInput["Domain"]["Media"][4]["mus"] = 1e-4
        else:
            self.mcxInput["Domain"]["Media"][4]["mua"] = 0
            self.mcxInput["Domain"]["Media"][4]["mus"] = 10.8
        # 5: Fat
        self.mcxInput["Domain"]["Media"][5]["n"] = self.modelParameters["OptParam"]["Fat"]["n"]
        self.mcxInput["Domain"]["Media"][5]["g"] = self.modelParameters["OptParam"]["Fat"]["g"]
        self.mcxInput["Domain"]["Media"][5]["mua"] = 0
        self.mcxInput["Domain"]["Media"][5]["mus"] = 10.4
        # 6: Muscle
        self.mcxInput["Domain"]["Media"][6]["n"] = self.modelParameters["OptParam"]["Muscle"]["n"]
        self.mcxInput["Domain"]["Media"][6]["g"] = self.modelParameters["OptParam"]["Muscle"]["g"]
        self.mcxInput["Domain"]["Media"][6]["mua"] = 0
        self.mcxInput["Domain"]["Media"][6]["mus"] = 4.2
        # 7: IJV
        self.mcxInput["Domain"]["Media"][7]["n"] = self.modelParameters["OptParam"]["IJV"]["n"]
        self.mcxInput["Domain"]["Media"][7]["g"] = self.modelParameters["OptParam"]["IJV"]["g"]
        self.mcxInput["Domain"]["Media"][7]["mua"] = 0
        self.mcxInput["Domain"]["Media"][7]["mus"] = 119.7
        # 8: CCA
        self.mcxInput["Domain"]["Media"][8]["n"] = self.modelParameters["OptParam"]["CCA"]["n"]
        self.mcxInput["Domain"]["Media"][8]["g"] = self.modelParameters["OptParam"]["CCA"]["g"]
        self.mcxInput["Domain"]["Media"][8]["mua"] = 0
        self.mcxInput["Domain"]["Media"][8]["mus"] = 119.7

    
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


    def setShapesFromUltrasound(self):
        # read ultrasound image segmentation parameters
        subject = self.modelParameters["GeoParam"]["subject"]
        date = self.modelParameters["GeoParam"]["date"]
        state = self.modelParameters["GeoParam"]["state"]
        currentDir = os.getcwd()
        os.chdir("../")
        with open("ultrasound_image_processing/blood_vessel_segmentation_line.json") as f:
            paramSet = json.load(f)[subject][date][state]
        os.chdir(currentDir)
        
        # extract parameters
        skinStartEdge = paramSet["skinStartEdge"]
        bloodThold = paramSet["bloodThold"]
        length10mmEdge = paramSet["length10mmEdge"]
        gridNumIn10mm = int(10/self.config["VoxelSize"])
        
        # load original scanned image
        currentDir = os.getcwd()
        os.chdir("../")
        image = cv2.imread("ultrasound_image_processing/{}_{}_{}.jpg".format(subject, date, state), cv2.IMREAD_GRAYSCALE)
        os.chdir(currentDir)
        # trim image to start from skin surface
        image = image[skinStartEdge:1500]
        
        # catch real target tissue area
        legalRowSet = {}
        legalColSet = {}
        for tissue in ["skin", "fat", "ijv", "cca"]:            
            if tissue == "ijv" or tissue == "cca":
                # catch target tissue
                # mesh coordinate
                coordinates = np.meshgrid(np.arange(image.shape[1]), np.flip(np.arange(image.shape[1])))
                coordinates = np.insert(coordinates, 0, 1, axis=0)  # add dummy coordinate
                # sketch potential region
                targetMatch = np.ones((image.shape[1], image.shape[1]), dtype=bool)
                # match line
                if tissue == "ijv":
                    for vector in np.array(paramSet[tissue]["v"]):
                        if vector[0] == -1:
                            targetMatch = targetMatch & ((vector.reshape(-1, 1, 1)[1:] * coordinates).sum(axis=0) < 0)
                        if vector[0] == 1:
                            targetMatch = targetMatch & ((vector.reshape(-1, 1, 1)[1:] * coordinates).sum(axis=0) > 0)
                    # match blood
                    targetMatch = targetMatch & (np.rot90(image[:image.shape[1], :image.shape[1]]) < bloodThold)
                if tissue == "cca":
                    targetMatch = targetMatch & ((coordinates[1]-paramSet[tissue]["v"][1])**2 + (coordinates[2]-paramSet[tissue]["v"][0])**2 <= paramSet[tissue]["v"][2]**2)
                
                # scaling
                scalePercentage = gridNumIn10mm / (length10mmEdge[1]-length10mmEdge[0])
                targetMatch = targetMatch.astype(float)
                targetMatch = cv2.resize(targetMatch, (int(np.round(targetMatch.shape[1]*scalePercentage)), int(np.round(targetMatch.shape[0]*scalePercentage))), interpolation=cv2.INTER_AREA)
                targetMatch = targetMatch.astype(bool)
                legalRow, legalCol = np.where(targetMatch == True)
                legalRow = targetMatch.shape[0]-legalRow
                
                legalRowSet[tissue] = legalRow
                legalColSet[tissue] = legalCol
        
        ### make volume
        # model
        modelX = self.mcxInput["Domain"]["Dim"][0]
        modelY = self.mcxInput["Domain"]["Dim"][1]
        modelZ = self.mcxInput["Domain"]["Dim"][2]
        # source
        srcHolderX = self.convertUnit(self.modelParameters["HardwareParam"]["Source"]["Holder"]["XSize"])
        srcHolderY = self.convertUnit(self.modelParameters["HardwareParam"]["Source"]["Holder"]["YSize"])
        srcHolderZ = self.convertUnit(self.modelParameters["HardwareParam"]["Source"]["Holder"]["ZSize"])
        irraWinRadius = self.convertUnit(self.modelParameters["HardwareParam"]["Source"]["Holder"]["IrraWinRadius"])
        # detecotr
        detHolderX = self.convertUnit(self.modelParameters["HardwareParam"]["Detector"]["Holder"]["XSize"])
        detHolderY = self.convertUnit(self.modelParameters["HardwareParam"]["Detector"]["Holder"]["YSize"])
        detHolderZ = self.convertUnit(self.modelParameters["HardwareParam"]["Detector"]["Holder"]["ZSize"])
        prismY = self.convertUnit(self.modelParameters["HardwareParam"]["Detector"]["Prism"]["YSize"])
        prismZ = self.convertUnit(self.modelParameters["HardwareParam"]["Detector"]["Prism"]["ZSize"])
        # start to construct !!
        # model and air (in the beginning)
        vol = np.ones((int(modelX), int(modelY), int(modelZ)))
        # source
        vol[int(modelX//2-srcHolderX//2):int(modelX//2+srcHolderX//2), 
            int(modelY//2-srcHolderY//2):int(modelY//2+srcHolderY//2),
            :int(srcHolderZ)] = 2  # holder
        for x in range(int(modelX//2)-int(np.ceil(irraWinRadius)), int(modelX//2)+int(np.ceil(irraWinRadius))):
            for y in range(int(modelY//2)-int(np.ceil(irraWinRadius)), int(modelY//2)+int(np.ceil(irraWinRadius))):
                isDist1 = np.sqrt((modelX//2-x)**2 + (modelY//2-y)**2) < np.ceil(irraWinRadius)
                isDist2 = np.sqrt((modelX//2-(x+1))**2 + (modelY//2-y)**2) < np.ceil(irraWinRadius)
                isDist3 = np.sqrt((modelX//2-x)**2 + (modelY//2-(y+1))**2) < np.ceil(irraWinRadius)
                isDist4 = np.sqrt((modelX//2-(x+1))**2 + (modelY//2-(y+1))**2) < np.ceil(irraWinRadius)
                if isDist1 or isDist2 or isDist3 or isDist4:
                    vol[x][y] = 1  # air
        # detector
        vol[int(modelX//2+srcHolderX//2):int(modelX//2+srcHolderX//2+detHolderX), 
            int(modelY//2-detHolderY//2):int(modelY//2+detHolderY//2),
            :int(detHolderZ)] = 2  # first holder
        vol[int(modelX//2-srcHolderX//2-detHolderX):int(modelX//2-srcHolderX//2), 
            int(modelY//2-detHolderY//2):int(modelY//2+detHolderY//2),
            :int(detHolderZ)] = 2  # second holder
        vol[int(modelX//2+srcHolderX//2):int(modelX//2+srcHolderX//2+detHolderX), 
            int(modelY//2-prismY//2):int(modelY//2+prismY//2),
            int(detHolderZ-prismZ):int(detHolderZ)] = 3  # first prism
        vol[int(modelX//2-srcHolderX//2-detHolderX):int(modelX//2-srcHolderX//2), 
            int(modelY//2-prismY//2):int(modelY//2+prismY//2),
            int(detHolderZ-prismZ):int(detHolderZ)] = 3  # second prism
        vol[int(modelX//2+srcHolderX//2):int(modelX//2+srcHolderX//2+detHolderX), 
            int(modelY//2-prismY//2):int(modelY//2+prismY//2),
            :int(detHolderZ-prismZ)] = 0  # first fiber
        vol[int(modelX//2-srcHolderX//2-detHolderX):int(modelX//2-srcHolderX//2), 
            int(modelY//2-prismY//2):int(modelY//2+prismY//2),
            :int(detHolderZ-prismZ)] = 0  # second fiber
        # muscle
        vol[:, :, int(detHolderZ):] = 6
        # fat
        fatDepth = int(paramSet["fat"]["x"]*scalePercentage)
        vol[:, :, int(detHolderZ):int(detHolderZ)+fatDepth] = 5
        # skin
        skinDepth = int(paramSet["skin"]["x"]*scalePercentage)
        vol[:, :, int(detHolderZ):int(detHolderZ)+skinDepth] = 4
        # ijv
        shiftNumber = np.round(modelY//2 - np.mean(legalRowSet["ijv"]), 0).astype(int)
        vol[:, np.array(legalRowSet["ijv"])+shiftNumber, np.array(legalColSet["ijv"])+int(detHolderZ)] = 7
        # cca
        vol[:, np.array(legalRowSet["cca"])+shiftNumber, np.array(legalColSet["cca"])+int(detHolderZ)] = 8
        
        # save and assign vol to mcxInput["Shapes"]
        vol = vol.astype(np.uint8)        
        sio.savemat(os.path.join(self.json_output, "tissue_model.mat"), {'tissue_model': vol})
        self.mcxInput["Shapes"] = vol.astype(np.uint8)


    def setOptodes(self):
        # detector (help to extend to left and right)
        modelX = self.mcxInput["Domain"]["Dim"][0]
        modelY = self.mcxInput["Domain"]["Dim"][1]
        detHolderZ = self.convertUnit(self.modelParameters["HardwareParam"]["Detector"]["Holder"]["ZSize"])
        prismZ = self.convertUnit(self.modelParameters["HardwareParam"]["Detector"]["Prism"]["ZSize"])
        for fiber in self.modelParameters["HardwareParam"]["Detector"]["Fiber"]:
            # right - bottom
            self.mcxInput["Optode"]["Detector"].append({"R": self.convertUnit(fiber["Radius"]),
                                                        "Pos": [modelX/2 + self.convertUnit(fiber["SDS"]),
                                                                modelY/2 - 2*self.convertUnit(fiber["Radius"]),
                                                                detHolderZ - prismZ
                                                                ]
                                                        })
            # left - bottom
            self.mcxInput["Optode"]["Detector"].append({"R": self.convertUnit(fiber["Radius"]),
                                                        "Pos": [modelX/2 - self.convertUnit(fiber["SDS"]),
                                                                modelY/2 - 2*self.convertUnit(fiber["Radius"]),
                                                                detHolderZ - prismZ
                                                                ]
                                                        })
            # right - middle (original)
            self.mcxInput["Optode"]["Detector"].append({"R": self.convertUnit(fiber["Radius"]),
                                                        "Pos": [modelX/2 + self.convertUnit(fiber["SDS"]),
                                                                modelY/2,
                                                                detHolderZ - prismZ
                                                                ]
                                                        })
            # left - middle
            self.mcxInput["Optode"]["Detector"].append({"R": self.convertUnit(fiber["Radius"]),
                                                        "Pos": [modelX/2 - self.convertUnit(fiber["SDS"]),
                                                                modelY/2,
                                                                detHolderZ - prismZ
                                                                ]
                                                        })
            # right - top
            self.mcxInput["Optode"]["Detector"].append({"R": self.convertUnit(fiber["Radius"]),
                                                        "Pos": [modelX/2 + self.convertUnit(fiber["SDS"]),
                                                                modelY/2 + 2*self.convertUnit(fiber["Radius"]),
                                                                detHolderZ - prismZ
                                                                ]
                                                        })
            # left - top
            self.mcxInput["Optode"]["Detector"].append({"R": self.convertUnit(fiber["Radius"]),
                                                        "Pos": [modelX/2 - self.convertUnit(fiber["SDS"]),
                                                                modelY/2 + 2*self.convertUnit(fiber["Radius"]),
                                                                detHolderZ - prismZ
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
        currentDir = os.getcwd()
        os.chdir("../")
        ledProfileIn3D = np.genfromtxt(self.modelParameters["HardwareParam"]["Source"]["Beam"]["ProfilePath"], delimiter=",")
        os.chdir(currentDir)
        angle = ledProfileIn3D[:, 0]
        cdf = np.cumsum(ledProfileIn3D[:, 1])
        inversecdf = PchipInterpolator(cdf, angle)        
        samplingSeeds = np.linspace(0, 1, num=int(self.modelParameters["HardwareParam"]["Source"]["LED"]["SamplingNumOfRadiationPattern"]))
        samplingAngles = inversecdf(samplingSeeds)
        # set source position
        self.mcxInput["Optode"]["Source"]["Pos"] = [modelX/2,
                                                    modelY/2,
                                                    self.convertUnit(self.modelParameters["HardwareParam"]["Source"]["Holder"]["ZSize"])-1e-4  # need to minus a very small value here. (if not, irradiating distribution will disappear)
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
    sessionID = "test"
    
    # initialize
    simulator = MCX(sessionID)
    
    # run forward mcx
    simulator.run()



