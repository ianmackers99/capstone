###
##
## This script uses Bayesian Optimisation with a UCB or EI acquisition function to select the next sample point for the ICL ML/AI captone project
## 
## To use this code:
##      1. set the 'week' variable to be this week of the capstone project
##      2. update the numpy input and output files with last week's queries and results and place them in a /data folder
##      3. set the 'xfile' and 'yfile' variables to point to that /data folder
##      4. set the 'functionID' variable to be the value of the function number (1 through 8) you wish to generate a sample point for
##      5. in the function specific parameters:
##          - set the type of acquisition function you wish to use
##          - set the hyperparameters of the acquisition function (Kappa or Xi)
##
## Please refer to 'Capstone Final Code Notebook.ipynb' for a full description of how this script functions
##
###
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.optimize import minimize
from scipy.stats import norm
import matplotlib.pyplot as plt

# 1. Set global variables
functionID = 8
week = 4
np.random.seed(42)
verbose = False

# Set input files
xfile = "C:/users/ianma/module0_capstone/week_" + str(week) + "/data/f" + str(functionID) + "_w" + str(week) + "_inputs.npy"
yfile = "C:/users/ianma/module0_capstone/week_" + str(week) + "/data/f" + str(functionID) + "_w" + str(week) + "_outputs.npy"  


#
# 2. Set function-specific parameters inc bounds, dimensions and acquisition variables
#
# Note that if xi is 0.1 it will explore more than 0.01
#
if functionID == 1:
    aqFunc = 'ei'
    kappa = 20
    xi = 0.1
    bounds = np.array([[0, 0], [1, 1]])
    dimensions = 2
#
if functionID == 2:
    aqFunc = 'ei'
    kappa = 20
    xi = 0.1
    bounds = np.array([[0, 0], [1, 1]])
    dimensions = 2
#
if functionID == 3:
    aqFunc = 'ei'
    kappa = 20
    xi = 0.1
    bounds = np.array([[0, 0, 0], [1, 1, 1]])
    dimensions = 3
#
if functionID == 4:
    aqFunc = 'ei'
    kappa = 20
    xi = 0.1
    bounds = np.array([[0, 0, 0, 0], [1, 1, 1, 1]])
    dimensions = 4
#
if functionID == 5:
    aqFunc = 'ei'
    kappa = 20
    xi = 0.1
    bounds = np.array([[0, 0, 0, 0], [1, 1, 1, 1]])
    dimensions = 4
#
if functionID == 6:
    aqFunc = 'ei'
    kappa = 20
    xi = 0.1
    bounds = np.array([[0, 0, 0, 0, 0], [1, 1, 1, 1, 1]])
    dimensions = 5
#
if functionID == 7:
    aqFunc = 'ei'
    kappa = 20
    xi = 0.1
    bounds = np.array([[0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1]])
    dimensions = 6
#
if functionID == 8:
    aqFunc = 'ei'
    kappa = 20
    xi = 0.1
    bounds = np.array([[0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1]])
    dimensions = 8


# 2. Import data
X = np.load(xfile)
Y = np.load(yfile)
Y_best = Y.max()


# 3. Fit the Gaussian Process
kernel = ConstantKernel(1.0) * RBF(length_scale=0.1)
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
gp.fit(X, Y)


# 4a. Upper Confidence Bound (UCB) Acquisition Function
def ucbAcquisition(X_cand, gp, kappa):
    mean, std = gp.predict(X_cand, return_std=True)
    return mean + kappa * std  # We want to maximize this value


# 4b. Expecting Improvement (EI) Acquisition Function
def eiAcquisition(X_cand, gp, Y_best, xi):
    mean, std = gp.predict(X_cand, return_std=True)
    with np.errstate(divide='warn'):
        improvement = mean - Y_best - xi
        Z = improvement / std
        ei = improvement * norm.cdf(Z) + std * norm.pdf(Z)
        ei[std == 0.0] = 0.0
    return ei


# 5. Propose Next Point function
def propose_next_point(gp, bounds, dimensions, n_candidates=10000):
    # Randomly sample candidates in nD space
    candidates = np.random.uniform(bounds[0], bounds[1], size=(n_candidates, dimensions))
    # Call the relevant acquisition function
    if aqFunc == 'ucb':
        scores = ucbAcquisition(candidates, gp, kappa)
        if verbose:
            print('\nAq Function: UCB', '\nY_best:', Y_best, '\nKappa:',  kappa)
    if aqFunc == 'ei':
        if verbose:
            print('\nAq Function: EI', '\nY_best:', Y_best, '\nXi:',  xi)
        scores = eiAcquisition(candidates, gp, Y_best, xi)
        
    # Return the candidate with the highest score
    next_point = candidates[np.argmax(scores)]
    
    if verbose:
        print('Winning candidate number: ', np.argmax(scores))
        print('Next point pre-decimal adjustment:', next_point)

    next_point = np.round(next_point,6)
    return next_point


##
##
# 6. Main function - get the next sample point
##
##
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