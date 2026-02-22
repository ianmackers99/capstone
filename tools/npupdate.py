#
# NP file updater. Set week, function and whether input or outfile variables to be updated
#
import numpy as np

lastweek = 3
thisweek = 4 # i.e. if we want to create the files for Capstone Week 2, set this to 2
functionID = 8
inputs = True
outputs = True

if inputs:
    # Open files
    xinfile = "C:/users/ianma/module0_capstone/week_" + str(lastweek) + "/data/f" + str(functionID) + "_w" + str(lastweek) + "_inputs.npy"
    xoutfile = "C:/users/ianma/module0_capstone/week_" + str(thisweek) + "/data/f" + str(functionID) + "_w" + str(thisweek) + "_inputs.npy"

    # Load last week's data 
    X = np.load(xinfile)
    print (X)

    # Last week's predictions
    if functionID == 1:
        newrow = np.array([0.611853,0.139494], dtype=np.float64)
    if functionID == 2:
        newrow = np.array([0.760808,0.850780], dtype=np.float64)
    if functionID == 3:
        newrow = np.array([0.986630,0.965119,0.004940], dtype=np.float64)
    if functionID == 4:
        newrow = np.array([0.404655,0.429048,0.226770,0.455883], dtype=np.float64)
    if functionID == 5:
        newrow = np.array([0.532623,0.594318,0.990109,0.985016], dtype=np.float64)
    if functionID == 6:
        newrow = np.array([0.315938,0.017065,0.899154,0.923549,0.050946], dtype=np.float64)
    if functionID == 7:
        newrow = np.array([0.339972,0.078939,0.442798,0.261049,0.343387,0.834572], dtype=np.float64)
    if functionID == 8:
        newrow = np.array([0.163797,0.350258,0.038865,0.166452,0.731353,0.649134,0.312781,0.551162], dtype=np.float64)

    Xupdated = np.vstack((X, newrow))

    print('Updated X array: ', Xupdated)

    np.save(xoutfile, Xupdated) 



##
# Output variables
##
if outputs:
    # Open files
    yinfile = "C:/users/ianma/module0_capstone/week_" + str(lastweek) + "/data/f" + str(functionID) + "_w" + str(lastweek) + "_outputs.npy"
    youtfile = "C:/users/ianma/module0_capstone/week_" + str(thisweek) + "/data/f" + str(functionID) + "_w" + str(thisweek) + "_outputs.npy"

    Y = np.load(yinfile)
    print (Y)

    # Last week's results
    if functionID == 1:
        newrowres = np.array([2.4247688654824617e-82], dtype=np.float64)
    if functionID == 2:
        newrowres = np.array([0.3686143841176872], dtype=np.float64)
    if functionID == 3:
        newrowres = np.array([-0.11861332481955808], dtype=np.float64)
    if functionID == 4:
        newrowres = np.array([-2.6353774299788344], dtype=np.float64)
    if functionID == 5:
        newrowres = np.array([1880.1864969021228], dtype=np.float64)
    if functionID == 6:
        newrowres = np.array([-0.9160700151162905], dtype=np.float64)
    if functionID == 7:
        newrowres = np.array([2.3480266237247505], dtype=np.float64)
    if functionID == 8:
        newrowres = np.array([9.8762944078321], dtype=np.float64)

    Yupdated = np.append(Y, newrowres)
    
    print('Updated Y array: ', Yupdated)

    np.save(youtfile, Yupdated) 


print('Done')

