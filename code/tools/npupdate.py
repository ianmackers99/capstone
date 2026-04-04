#
# NP file updater. Set week, function and whether input or outfile variables to be updated
#
import numpy as np

lastweek = 9
thisweek = 10 # i.e. if we want to create the files for Capstone Week 2, set this to 2
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

    ##
    ## Last week's predictions
    ##
    if functionID == 1:
        newrow = np.array([0.740000,0.740000], dtype=np.float64)
    if functionID == 2:
        newrow = np.array([0.720000,0.950000], dtype=np.float64)
    if functionID == 3:
        newrow = np.array([0.420000,0.450000,0.500000], dtype=np.float64)
    if functionID == 4:
        newrow = np.array([0.360000,0.420000,0.300000,0.460000], dtype=np.float64)
    if functionID == 5:
        newrow = np.array([0.620000,0.720000,0.995000,0.970000], dtype=np.float64)
    if functionID == 6:
        newrow = np.array([0.450000,0.380000,0.580000,0.750000,0.150000], dtype=np.float64)
    if functionID == 7:
        newrow = np.array([00.340000,0.120000,0.410000,0.260000,0.360000,0.810000], dtype=np.float64)
    if functionID == 8:
        newrow = np.array([0.020000,0.150000,0.080000,0.050000,0.920000,0.980000,0.030000,0.120000], dtype=np.float64)

    Xupdated = np.vstack((X, newrow))

    print('Updated X array: ', Xupdated)

    np.save(xoutfile, Xupdated) 



##
## Last week's results
##
if outputs:
    # Open files
    yinfile = "C:/users/ianma/module0_capstone/week_" + str(lastweek) + "/data/f" + str(functionID) + "_w" + str(lastweek) + "_outputs.npy"
    youtfile = "C:/users/ianma/module0_capstone/week_" + str(thisweek) + "/data/f" + str(functionID) + "_w" + str(thisweek) + "_outputs.npy"

    Y = np.load(yinfile)
    print (Y)

    # Last week's results
    if functionID == 1:
        newrowres = np.array([6.854713532414845e-19], dtype=np.float64)
    if functionID == 2:
        newrowres = np.array([0.6570567569059171], dtype=np.float64)
    if functionID == 3:
        newrowres = np.array([-0.030318346224121093], dtype=np.float64)
    if functionID == 4:
        newrowres = np.array([-1.2283180410904389], dtype=np.float64)
    if functionID == 5:
        newrowres = np.array([2308.98059899065], dtype=np.float64)
    if functionID == 6:
        newrowres = np.array([-0.09087030291643944], dtype=np.float64)
    if functionID == 7:
        newrowres = np.array([2.467619976670761], dtype=np.float64)
    if functionID == 8:
        newrowres = np.array([9.65126], dtype=np.float64)

    Yupdated = np.append(Y, newrowres)
    
    print('Updated Y array: ', Yupdated)

    np.save(youtfile, Yupdated) 


print('Done')

