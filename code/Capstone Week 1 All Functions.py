###
##
## This script uses Bayesian Optimisation with a UCB or EI acquisition function to select the next sample point for the ICL ML/AI captone project
## 
## To use this code:
##      1. place the numpy input and output files in a /data folder (for Week 1, these will be the initial dataset provided by the faculty)
##          - the input vaiable file must be called "f<function #>_w<week #>_input.npy" i.e. f1_w1_inputs.npy
##          - the output vaiable file must be called "f<function #>_w<week #>_output.npy" i.e. f1_w1_outputs.npy
##      2. In the global variables:
##          - set the 'week' variable to be this week of the capstone project
##          - set the 'functionID' variable to be the value of the function number (1 through 8) you wish to generate a sample point for
##          - ensure the path in the 'xfile' and 'yfile' variables to points to your /data folder
##
##
## Please refer to 'Capstone Final Code Notebook.ipynb' for a full description of how this script functions
##
###
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# 1. Set global variables
functionID = 1
week = 1
np.random.seed(42)


# Set input files
xfile = "C:/users/ianma/module0_capstone/week_" + str(week) + "/data/f" + str(functionID) + "_w" + str(week) + "_inputs.npy"
yfile = "C:/users/ianma/module0_capstone/week_" + str(week) + "/data/f" + str(functionID) + "_w" + str(week) + "_outputs.npy"  

#
# 2. Set function-specific parameters inc space bounds and dimensions
#
if functionID == 1 or functionID == 2:
    bounds = np.array([[0, 0], [1, 1]])
    dimensions = 2
if functionID == 3:
    bounds = np.array([[0, 0, 0], [1, 1, 1]])
    dimensions = 3
if functionID == 4 or functionID == 5:
    bounds = np.array([[0, 0, 0, 0], [1, 1, 1, 1]])
    dimensions = 4
if functionID == 6:
    bounds = np.array([[0, 0, 0, 0, 0], [1, 1, 1, 1, 1]])
    dimensions = 5
if functionID == 7:
    bounds = np.array([[0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1]])
    dimensions = 6
if functionID == 8:
    bounds = np.array([[0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1]])
    dimensions = 8


# 2. Import data
X = np.load(xfile)
Y = np.load(yfile)


# 3. Fit the Gaussian Process
kernel = ConstantKernel(1.0) * RBF(length_scale=0.1)
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
gp.fit(X, Y)


# 4. Upper Confidence Bound (UCB) Acquisition Function
def ucbAcquisition(X_cand, gp, kappa=10):
    mean, std = gp.predict(X_cand, return_std=True)
    return mean + kappa * std  # We want to maximize this value


# 5. Propose Next Point function - based on the max acquisition value
def propose_next_point(gp, bounds, dimensions, n_candidates=10000):
    # Randomly sample candidates in nD space
    candidates = np.random.uniform(bounds[0], bounds[1], size=(n_candidates, dimensions))
    # Evaluate candidates
    scores = ucbAcquisition(candidates, gp)
    # Return candidate with highest score
    next_point = candidates[np.argmax(scores)]
    next_point = np.round(next_point,6)
    return next_point


# 6. Main function - fetch the next sample point
next_point = propose_next_point(gp, bounds, dimensions)


#7. Create the query string for submission
columns= len(next_point)
colcount = 0
query=''
packzero = ''
while colcount < columns:
    element = str(round(next_point[colcount], 6))
    strlen = len(element)
    if strlen < 8:
        packlen = 8 - strlen
        while len(packzero) < packlen:
            packzero = packzero + '0'
        element = element + packzero
    query = query + '-' + element
    colcount = colcount + 1


#8. Print the query string
print(f"Function {functionID}")
print (query[1:])