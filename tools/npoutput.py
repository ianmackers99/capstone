#
# NP file loader
#
import numpy as np


# 1. Set global variables
week = 4
functionID = 8
inputs = True
outputs = True

## 
# Input variables
##
if inputs:
        xinfile = "C:/users/ianma/module0_capstone/week_" + str(week) + "/data/f" + str(functionID) + "_w" + str(week) + "_inputs.npy"
        xoutfile = "C:/users/ianma/module0_capstone/week_" + str(week) + "/data/f" + str(functionID) + "_w" + str(week) + "_inputs.txt"
        X = np.load(xinfile)
        file1 = open (xoutfile, 'w')
        writecount = 0
        
        for row in X:
            row = str(row) + "\n"
            file1.write(row)
            writecount = writecount + 1
        print(writecount, 'records written to', xoutfile)
        print('Closing all files...')
        file1.close()
    
## 
# Output variables
##
if outputs:
        yinfile = "C:/users/ianma/module0_capstone/week_" + str(week) + "/data/f" + str(functionID) + "_w" + str(week) + "_outputs.npy"
        youtfile = "C:/users/ianma/module0_capstone/week_" + str(week) + "/data/f" + str(functionID) + "_w" + str(week) + "_outputs.txt"
        Y = np.load(yinfile)
        file2 = open (youtfile, 'w')
        ywritecount = 0
        for yrow in Y:
                yrow = str(yrow) + "\n"
                file2.write(yrow)
                ywritecount = ywritecount + 1
        print(ywritecount, 'records written to', youtfile)
        print('Closing all files...')
        file2.close()
    
print('Done')
