import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

# Load handwiring data
textfolderpath = "../../../text/text_normalized"
textfolder = os.listdir(textfolderpath)
textdict = {}
alldata = []
for file in textfolder:
    if file.endswith(".csv"):
        textlabel = file.split(".")[0]
        
        # 2*800
        textdata = pd.read_csv(textfolderpath + "/" + file)

        # Append 135 on each side to make the data equally long, also normalize to be between -1,1
        #Convert to np
        textdata = textdata.to_numpy()
        textdata = np.concatenate((np.zeros([85,2]), textdata, np.zeros([85,2])), axis=0)/600

        alldata.append(textdata)
resultfolder = "./digits=10"
settingsfolders = os.listdir(resultfolder)



for settingsfolder in settingsfolders:
    if settingsfolder == "lrh=1e-05_lro=0.0001_ami=1.0_ama=10.0":
        datafolder = resultfolder + "/" + settingsfolder 
        datafiles = os.listdir(datafolder)

        totalerror = 0
        for datafile in datafiles:
            if datafile.endswith("filter.txt"):
                data = np.loadtxt(datafolder + "/" + datafile)
                summed = np.sum(data, axis=0)
                errors = np.zeros(len(alldata))
                for idx,dat in enumerate(alldata):
                    errors[idx] = np.sum((summed - dat[:,0])**2)
                
                lowest = np.argmin(errors)
                plt.plot(alldata[lowest][:,0], label="target")
                plt.plot(summed, label="output")
                plt.legend()
                totalerror += errors[lowest]
                plt.show()
                plt.clf()
        print(settingsfolder + f"has total error of {totalerror}")