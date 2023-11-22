# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 14:06:13 2021

@author: EricSyu
"""

# from IPython import get_ipython
# get_ipython().magic('clear')
# get_ipython().magic('reset -f')
import matplotlib.pyplot as plt
plt.close("all")
import numpy as np
from scipy.interpolate import PchipInterpolator
import jdata as jd
from glob import glob
import os
import sys
import json

class MCX:
    
    # Initialize
    def __init__(self, sessionID, wmc=True):
        # load session ID
        self.sessionID = sessionID
        
        # load config and related file
        with open(os.path.join(self.sessionID, "config.json")) as f:
            self.config = json.load(f)
        with open(self.config["MCXInputPath"]) as f:
            self.mcxInput = json.load(f)
        
        # load model parameters        
        with open(os.path.join(self.sessionID, "model_parameters.json")) as f:
            self.modelParameters = json.load(f)
        
        # create related folders in output path
        self.createFolder()
        
        # check if simulation_result.json is existed
        resultSet = glob(os.path.join(self.post_analysis, "*.json"))
        if len(resultSet) == 0:
            self.createSimResultTemplate()
        
        # check if mcx input file is existed
        if not os.path.isfile(os.path.join(self.json_output, "input_{}.json".format(self.sessionID))):
            self.makeMCXInput(wmc=wmc)


    def createFolder(self):
        # set folder path        
        # main-path
        self.session = os.path.join(self.config["OutputPath"], self.sessionID)        
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
    
    
    def createSimResultTemplate(self):
        fiberSet = self.modelParameters["HardwareParam"]["Detector"]["Fiber"]
        if len(fiberSet) > 0:
            result = {
                "SessionID:": self.sessionID,
                "AnalyzedSampleNum": 0,
                "GroupingNum": 0,
                "PhotonNum": {"RawSample": "{:.4e}".format(self.config["PhotonNum"]), "GroupingSample": 0},
                # consider moving-average, detetor number reduce by 2.
                "MovingAverageGroupingSampleValues": {"sds_{}".format(fiberSet[idx+1]["SDS"]):np.zeros(10, dtype=int).tolist() for idx in range(len(fiberSet)-2)},
                "MovingAverageGroupingSampleStd": {"sds_{}".format(fiberSet[idx+1]["SDS"]):None for idx in range(len(fiberSet)-2)},
                "MovingAverageGroupingSampleMean": {"sds_{}".format(fiberSet[idx+1]["SDS"]):None for idx in range(len(fiberSet)-2)},
                # the values here are just for initialization
                "MovingAverageGroupingSampleCV": {"sds_{}".format(fiberSet[idx+1]["SDS"]):1000+np.random.rand() for idx in range(len(fiberSet)-2)}
            }
            muaTypeSet = glob(os.path.join(self.sessionID, "mua*"))
            # print(f"muaTypeSet: {muaTypeSet}")
            for muaType in muaTypeSet:
                muaType = muaType.split("/")[-1][:-5]
                with open(os.path.join(self.post_analysis, f"{self.sessionID}_simulation_result_{muaType}.json"), "w") as f:
                    json.dump(result, f, indent=4)
    

    # Main function to run simulation by passing formal configuration to MCX bin.
    def run(self, simIdx, gpuIdx=1):
        # make command and run
        for customizedCommand in self.config["CustomizedCommands"]:
            command = self.getCommand(simIdx, customizedCommand, gpuIdx=gpuIdx)
            # run
            sys.stdout.flush()
            currentDir = os.getcwd()
            os.chdir(self.config["BinaryPath"])
            print("Current position to run MCX:\n", os.getcwd(), end="\n\n")
            print("Command sent to MCX:\n{}".format(command), end="\n\n")
            print("∎∎ Start to run # {} ...".format(simIdx), end=" ")
            os.system(command)
            print("Finished !! ∎∎", end="\n\n")
            os.chdir(currentDir)
        # remove .jnii
        if "--outputtype E" not in self.config["CustomizedCommands"][0]:
            jniiOutputPathSet = glob(os.path.abspath(os.path.join(self.mcx_output, "*.jnii")))
            for jniiOutputPath in jniiOutputPathSet:
                os.remove(jniiOutputPath)


    def replay(self, volDim, gpuIdx=1):
        detOutputPathSet = glob(os.path.join(self.mcx_output, "*.mch"))  # about paths of detected photon data
        detOutputPathSet.sort(key=lambda x: int(x.split("_")[-1].replace(".mch", "")))
        
        self.makeMCXInput(wmc=True, designatedVolDim=volDim)
        customizedCommand = self.config["CustomizedCommands"][1]
        for i in range(len(detOutputPathSet)):
            command = self.getCommand(i, customizedCommand, gpuIdx=gpuIdx)
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
    def getCommand(self, simOrder, customizedCommand, gpuIdx=1): 
        # basic setting
        sessionName = "{}_{}".format(self.sessionID, simOrder)
        geometryFile = os.path.abspath(os.path.join(self.json_output, "input_{}.json".format(self.sessionID)))
        root = os.path.join(os.path.abspath(self.session), "mcx_output")
        
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
        command += f"--gpu {gpuIdx} "  # use specific gpu, default "1"
        command += "--autopilot 1 " 
        command += "--photon {} ".format(self.config["PhotonNum"])
        command += "--repeat 1 "
        command += "--normalize 1 "
        command += "--bc aaaaar "  # allow reflection in -z surface (medium 0 and 1 surface)
        command += "--unitinmm {} ".format(self.config["VoxelSize"])
        command += "--skipradius -2 " 
        command += "--array 0 " 
        command += "--dumpmask 0 "
        command += "--maxdetphoton {} ".format(4e7)
        command += "--srcfrom0 1 "
        command += "--debug P "
        if "mch" in customizedCommand:  # for replay
            customizedCommand = customizedCommand.replace("mch", os.path.abspath(os.path.join(self.mcx_output, "{}.mch".format(sessionName))))
        command += customizedCommand

        return command


    def makeMCXInput(self, wmc=True, designatedVolDim=None):
        """

        Parameters
        ----------
        designatedVolDim : TYPE, optional
            If designatedVolDim is given, its form should be [XSize, YSize, ZSize] in mm.

        Returns
        -------
        None.

        """
        # set Session
        self.mcxInput["Session"]["ID"] = self.sessionID
        
        # set Domain Media (optical properties of tissue model)
        self.setDomainMedia(wmc=wmc)        
        
        # set Domain Dim
        if designatedVolDim:
            self.mcxInput["Domain"]["Dim"] = [int(self.convertUnit(designatedVolDim[0])),
                                              int(self.convertUnit(designatedVolDim[1])),
                                              int(self.convertUnit(designatedVolDim[2]))
                                              ]
        else:
            vol = np.load(self.config["VolumePath"])
            self.mcxInput["Domain"]["Dim"] = list(vol.shape)
        
        # set Domain OriginType
        self.mcxInput["Domain"]["OriginType"] = 1
        
        # set Shapes
        self.mcxInput["Shapes"] = vol
        
        # set Optodes
        self.setOptodes()
        
        # save mcxInput to output/json_output
        jd.save(jd.encode(self.mcxInput, {'compression':'zlib','base64':1}), 
                os.path.join(self.json_output, "input_{}.json".format(self.sessionID)))
        self.mcxInput["Optode"]["Source"] = {"Type": "pencil", 
                                             "Pos": self.mcxInput["Optode"]["Source"]["Pos"],
                                             "Dir": [0.0, 0.0, 1.0]}  # change to pencil, for preview
        self.mcxInput["Shapes"] = self.mcxInput["Shapes"][:, :, :28]  # slice volume, for preview
        jd.save(jd.encode(self.mcxInput, {'compression':'zlib','base64':1}), 
                os.path.join(self.json_output, "input_{}_forpreview.json".format(self.sessionID)))


    def setDomainMedia(self, wmc=True):
        if self.config["Type"] == "ijv":
            # if wmc is False, read mua (ex: run fluence data)
            if wmc is not True:
                muaPath = glob(os.path.join(self.sessionID, "mua*"))[0]
                with open(muaPath) as f:
                    mua = json.load(f)
            
            # 0: Fiber (mua always 0)
            self.mcxInput["Domain"]["Media"][0]["n"] = self.modelParameters["OptParam"]["Fiber"]["n"]
            self.mcxInput["Domain"]["Media"][0]["g"] = self.modelParameters["OptParam"]["Fiber"]["g"]
            self.mcxInput["Domain"]["Media"][0]["mua"] = 0
            self.mcxInput["Domain"]["Media"][0]["mus"] = self.modelParameters["OptParam"]["Fiber"]["mus"]
            
            # 1: Air (mua always 0)
            self.mcxInput["Domain"]["Media"][1]["n"] = self.modelParameters["OptParam"]["Air"]["n"]
            self.mcxInput["Domain"]["Media"][1]["g"] = self.modelParameters["OptParam"]["Air"]["g"]
            self.mcxInput["Domain"]["Media"][1]["mua"] = 0
            self.mcxInput["Domain"]["Media"][1]["mus"] = self.modelParameters["OptParam"]["Air"]["mus"]
            
            # 2: PLA
            self.mcxInput["Domain"]["Media"][2]["n"] = self.modelParameters["OptParam"]["PLA"]["n"]
            self.mcxInput["Domain"]["Media"][2]["g"] = self.modelParameters["OptParam"]["PLA"]["g"]
            self.mcxInput["Domain"]["Media"][2]["mua"] = 0 if wmc is True else mua["2: PLA"]
            self.mcxInput["Domain"]["Media"][2]["mus"] = self.modelParameters["OptParam"]["PLA"]["mus"]
            
            # 3: Prism (mua always 0)
            self.mcxInput["Domain"]["Media"][3]["n"] = self.modelParameters["OptParam"]["Prism"]["n"]
            self.mcxInput["Domain"]["Media"][3]["g"] = self.modelParameters["OptParam"]["Prism"]["g"]
            self.mcxInput["Domain"]["Media"][3]["mua"] = 0
            self.mcxInput["Domain"]["Media"][3]["mus"] = self.modelParameters["OptParam"]["Prism"]["mus"]
            
            # 4: Skin
            self.mcxInput["Domain"]["Media"][4]["n"] = self.modelParameters["OptParam"]["Skin"]["n"]
            self.mcxInput["Domain"]["Media"][4]["g"] = self.modelParameters["OptParam"]["Skin"]["g"]
            self.mcxInput["Domain"]["Media"][4]["mua"] = 0 if wmc is True else mua["4: Skin"]
            self.mcxInput["Domain"]["Media"][4]["mus"] = self.modelParameters["OptParam"]["Skin"]["mus"]
            # if "--save2pt 1" in str(self.config["CustomizedCommands"]):
            #     self.mcxInput["Domain"]["Media"][4]["mua"] = 4e6
            #     self.mcxInput["Domain"]["Media"][4]["mus"] = 1e-4
            # else:
            #     self.mcxInput["Domain"]["Media"][4]["mua"] = 0
            #     self.mcxInput["Domain"]["Media"][4]["mus"] = self.modelParameters["OptParam"]["Skin"]["mus"]
            
            # 5: Fat
            self.mcxInput["Domain"]["Media"][5]["n"] = self.modelParameters["OptParam"]["Fat"]["n"]
            self.mcxInput["Domain"]["Media"][5]["g"] = self.modelParameters["OptParam"]["Fat"]["g"]
            self.mcxInput["Domain"]["Media"][5]["mua"] = 0 if wmc is True else mua["5: Fat"]
            self.mcxInput["Domain"]["Media"][5]["mus"] = self.modelParameters["OptParam"]["Fat"]["mus"]
            
            # 6: Muscle
            self.mcxInput["Domain"]["Media"][6]["n"] = self.modelParameters["OptParam"]["Muscle"]["n"]
            self.mcxInput["Domain"]["Media"][6]["g"] = self.modelParameters["OptParam"]["Muscle"]["g"]
            self.mcxInput["Domain"]["Media"][6]["mua"] = 0 if wmc is True else mua["6: Muscle"]
            self.mcxInput["Domain"]["Media"][6]["mus"] = self.modelParameters["OptParam"]["Muscle"]["mus"]
            
            # 7: Muscle or IJV (Perturbed Region)
            if self.sessionID.split("_")[1] == "col":
                self.mcxInput["Domain"]["Media"][7]["n"] = self.modelParameters["OptParam"]["Muscle"]["n"]
                self.mcxInput["Domain"]["Media"][7]["g"] = self.modelParameters["OptParam"]["Muscle"]["g"]
                self.mcxInput["Domain"]["Media"][7]["mua"] = 0 if wmc is True else mua["7: Muscle or IJV (Perturbed Region)"]
                self.mcxInput["Domain"]["Media"][7]["mus"] = self.modelParameters["OptParam"]["Muscle"]["mus"]
            elif self.sessionID.split("_")[1] == "dis":
                self.mcxInput["Domain"]["Media"][7]["n"] = self.modelParameters["OptParam"]["IJV"]["n"]
                self.mcxInput["Domain"]["Media"][7]["g"] = self.modelParameters["OptParam"]["IJV"]["g"]
                self.mcxInput["Domain"]["Media"][7]["mua"] = 0 if wmc is True else mua["7: Muscle or IJV (Perturbed Region)"]
                self.mcxInput["Domain"]["Media"][7]["mus"] = self.modelParameters["OptParam"]["IJV"]["mus"]
            else:
                raise Exception("Something wrong in your config[VolumePath] !")
            
            # 8: IJV
            self.mcxInput["Domain"]["Media"][8]["n"] = self.modelParameters["OptParam"]["IJV"]["n"]
            self.mcxInput["Domain"]["Media"][8]["g"] = self.modelParameters["OptParam"]["IJV"]["g"]
            self.mcxInput["Domain"]["Media"][8]["mua"] = 0 if wmc is True else mua["8: IJV"]
            self.mcxInput["Domain"]["Media"][8]["mus"] = self.modelParameters["OptParam"]["IJV"]["mus"]
            
            # 9: CCA
            self.mcxInput["Domain"]["Media"][9]["n"] = self.modelParameters["OptParam"]["CCA"]["n"]
            self.mcxInput["Domain"]["Media"][9]["g"] = self.modelParameters["OptParam"]["CCA"]["g"]
            self.mcxInput["Domain"]["Media"][9]["mua"] = 0 if wmc is True else mua["9: CCA"]
            self.mcxInput["Domain"]["Media"][9]["mus"] = self.modelParameters["OptParam"]["CCA"]["mus"]
        
        if self.config["Type"] == "phantom":
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
            # 3: Prism
            self.mcxInput["Domain"]["Media"][3]["n"] = self.modelParameters["OptParam"]["Prism"]["n"]
            self.mcxInput["Domain"]["Media"][3]["g"] = self.modelParameters["OptParam"]["Prism"]["g"]
            self.mcxInput["Domain"]["Media"][3]["mua"] = 0
            self.mcxInput["Domain"]["Media"][3]["mus"] = self.modelParameters["OptParam"]["Prism"]["mus"]
            # 4: Phantom body
            self.mcxInput["Domain"]["Media"][4]["n"] = self.modelParameters["OptParam"]["Phantom body"]["n"]
            self.mcxInput["Domain"]["Media"][4]["g"] = self.modelParameters["OptParam"]["Phantom body"]["g"]
            if "--save2pt 1" in str(self.config["CustomizedCommands"]):
                self.mcxInput["Domain"]["Media"][4]["mua"] = 4e4
                self.mcxInput["Domain"]["Media"][4]["mus"] = 1e-4
            else:
                self.mcxInput["Domain"]["Media"][4]["mua"] = 0
                self.mcxInput["Domain"]["Media"][4]["mus"] = self.modelParameters["OptParam"]["Phantom body"]["mus"]


    def setOptodes(self):
        if self.modelParameters["HardwareParam"]["Source"]["Beam"]["Type"] == "anglepattern":
            if self.config["Type"] == "ijv":
                # detector (help to extend to left and right)
                modelX = self.mcxInput["Domain"]["Dim"][0]
                modelY = self.mcxInput["Domain"]["Dim"][1]
                detHolderZ = self.convertUnit(self.modelParameters["HardwareParam"]["Detector"]["Holder"]["ZSize"])
                prismZ = self.convertUnit(self.modelParameters["HardwareParam"]["Detector"]["Prism"]["legSize"])
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
                # source
                # set source position
                self.mcxInput["Optode"]["Source"]["Pos"] = [modelX/2,
                                                            modelY/2,
                                                            self.convertUnit(self.modelParameters["HardwareParam"]["Source"]["Holder"]["ZSize"])-1e-4  # need to minus a very small value here. (if not, irradiating distribution will disappear)
                                                            ]
            if self.config["Type"] == "phantom":
                # detector (help to extend to left and right)
                modelX = self.mcxInput["Domain"]["Dim"][0]
                srcHolderX = self.convertUnit(self.modelParameters["HardwareParam"]["Source"]["Holder"]["XSize"])
                srcCenterX = modelX//2-srcHolderX//2
                modelY = self.mcxInput["Domain"]["Dim"][1]
                detHolderZ = self.convertUnit(self.modelParameters["HardwareParam"]["Detector"]["Holder"]["ZSize"])
                prismZ = self.convertUnit(self.modelParameters["HardwareParam"]["Detector"]["Prism"]["legSize"])
                for fiber in self.modelParameters["HardwareParam"]["Detector"]["Fiber"]:
                    # right - bottom
                    self.mcxInput["Optode"]["Detector"].append({"R": self.convertUnit(fiber["Radius"]),
                                                                "Pos": [srcCenterX + self.convertUnit(fiber["SDS"]),
                                                                        modelY/2 - 2*self.convertUnit(fiber["Radius"]),
                                                                        detHolderZ - prismZ
                                                                        ]
                                                                })
                    # right - middle (original)
                    self.mcxInput["Optode"]["Detector"].append({"R": self.convertUnit(fiber["Radius"]),
                                                                "Pos": [srcCenterX + self.convertUnit(fiber["SDS"]),
                                                                        modelY/2,
                                                                        detHolderZ - prismZ
                                                                        ]
                                                                })
                    # right - top
                    self.mcxInput["Optode"]["Detector"].append({"R": self.convertUnit(fiber["Radius"]),
                                                                "Pos": [srcCenterX + self.convertUnit(fiber["SDS"]),
                                                                        modelY/2 + 2*self.convertUnit(fiber["Radius"]),
                                                                        detHolderZ - prismZ
                                                                        ]
                                                                })
                # source
                # set source position
                self.mcxInput["Optode"]["Source"]["Pos"] = [srcCenterX,
                                                            modelY/2,
                                                            self.convertUnit(self.modelParameters["HardwareParam"]["Source"]["Holder"]["ZSize"])-1e-4  # need to minus a very small value here. (if not, irradiating distribution will disappear)
                                                            ]
            
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
        
        elif self.modelParameters["HardwareParam"]["Source"]["Beam"]["Type"] == "cone":
            modelX = self.mcxInput["Domain"]["Dim"][0]
            modelY = self.mcxInput["Domain"]["Dim"][1]
            detHolderZ = self.convertUnit(self.modelParameters["HardwareParam"]["Detector"]["Holder"]["ZSize"])
            prismZ = self.convertUnit(self.modelParameters["HardwareParam"]["Detector"]["Prism"]["legSize"])
            
            self.mcxInput["Optode"]["Source"]["Type"] = self.modelParameters["HardwareParam"]["Source"]["Beam"]["Type"]
            self.mcxInput["Optode"]["Source"]["Pos"] = [modelX/2 + self.convertUnit(20),  # for adjoint method, set detector-like source in sds = 20 mm (middle of 3~40)
                                                        modelY/2,
                                                        detHolderZ - prismZ - 3.3192  # 3.3192 is the cone height
                                                        ]
            self.mcxInput["Optode"]["Source"]["Param1"] = [0.4169, 0, 0, 0]  # 0.4169 is half-angle of cone in radians
        
        else:
            raise Exception("Not acceptable source type !")
    
    
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
    
    # parameters
    sessionID = "test"
    
    # initialize
    simulator = MCX(sessionID)
    
    # run forward mcx
    simulator.run()



