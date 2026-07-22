import matplotlib.pyplot as plt
from pathlib import Path
import sys
from glob import glob
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))


def retrievedata(datadirectory, setting):

    lossfile = next(datadirectory.glob(f"losses_{setting}"), "Empty")
    targetfile = next(datadirectory.glob(f"target_{setting}"), "Empty")
    twitchhistfile = next(datadirectory.glob(f"twitchhistory_{setting}"), "Empty")

    if lossfile == "Empty":
        raise FileNotFoundError(f"No lossfile found corresponding to setting: {setting}")
    if targetfile == "Empty":
        raise FileNotFoundError(f"No targetfile found corresponding to setting: {setting}")
    if twitchhistfile == "Empty":
        raise FileNotFoundError(f"No twitchfile found corresponding to setting: {setting}")

        
    loss = np.loadtxt(f"{lossfile}")
    target = np.loadtxt(f"{targetfile}")
    twitchhist = np.loadtxt(f"{twitchhistfile}")

    return loss, target, twitchhist

def getuniqueoutputsettings(resultsdir):
    settings = []
    for item in resultsdir.iterdir():
        
        if item.is_file():
            filename = str(item).split("/")[-1]
            parts = filename.split("_")
            if len(parts) == 3: # sine, block
                setting =  parts[-2] + "_" + parts[-1] 
            elif len(parts) == 4: #gauss
                setting = parts[-3] + "_" + parts[-2] + "_" + parts[-1]
            else:
                raise ValueError(f"Unexpected number of splits in {filename}")

            settings.append(setting)

    return list(set(settings))



def plot(loss, target, twitchhist, filename, targetdir):

    _, (ax1, ax2) = plt.subplots(2, 1)

    # plot fit and learning curve
    ax1.plot(np.sum(twitchhist, axis=0), label="twitches")
    ax1.plot(target, label="target")
    ax1.legend()
    ax1.set_xlabel("x")
    ax1.set_ylabel("amplitude")
    
    ax2.plot(loss)
    ax2.set_xlabel("Iterations")
    ax2.set_ylabel("MSE")

    plt.savefig(targetdir / f"{str(filename).replace(".txt", ".png")}")
    plt.clf()




def main():
    # List result files
    TRAINRESULTS_DIR = ROOT_DIR / "data/basefunctions/trainresults/"
    OUTPUT_DIR = ROOT_DIR / "data/basefunctions/images/trainevaluation/"

    settings = getuniqueoutputsettings(TRAINRESULTS_DIR)

    for setting in settings: # prevent having to load all data at once
        loss, target, twitchhist = retrievedata(TRAINRESULTS_DIR, setting)
        plot(loss, target, twitchhist, setting, OUTPUT_DIR)
    



if __name__ == "__main__":
    main()