# Imperial College London - ML/AI Course - BBO Capstone Project
## Project Overview

The Black Box Optimisation (BBO) Capstone Project is a challenge designed to test a student’s knowledge and understanding of Machine Learning (ML) tools.

The overall goal of the capstone project is to find the maximum output values of 8 hidden black box functions. To achieve this, students are encouraged to use a variety of tools to determine the next best point to query the black box function. The results are fed back into the model each week so it progressively learns the shape of the functions.

This mimics a common, real-world scenario where data scientists are asked to make predictions based on limited initial information and then update those predictions as additional data is received.


## Navigating this repo

- /code - contains the code to run the model. Begin with **Capstone Final Code Notebook.ipynb**.
- /code/tools - contains tools to update the Numpy arrays each week and output them as plain text for review
- /data - contains the Numpy arrays for each week of the project
- /documentation - contains a **Datasheet for the dataset** and a **Model Card** for the model
- /queries and results - contains a spreadsheet with a tab showing the query and result for each function each week and the improvement over time


## Input and Outputs

At the beginning of the challenge, students are given a small, initial input and output dataset for each function in the form of Numpy arrays. The number of dimensions for each function varies between 2 and 8 and the number of initial samples varies between 10 and 40.

For example, the input for a 5-dimensional function with 5 features might look like this:

- 0.007279 0.922324 0.065996 0.938621 0.028153

And the corresponding output might look like this:

- 0.53899612

Students use these input and output values to create a Gaussian Process which acts as a surrogate for the true function, which is then used to simulate its behaviour.

Each week, students pass a large number of candidate sample points through the surrogate model and use an acquisition function to assess them and recommend a new, best sample point for the coming week. The sample points must be presented to the function as a sequence of hyphen-separated numbers in the range 0 and 1, rounded to 6 decimal places. Each number represents the value of a feature and together the numbers form a query which the black box will consume and then generate a result. A typical query for a 5-dimension function might look like this:

- 0.007279-0.922324-0.065996-0.938621-0.028153

The results returned by the black box might look like this:

- 1.06562362

Students add the new queries and corresponding results to the Numpy arrays each week and use them to train their model the following week.


## Challenge Objective

The overall goal of the capstone project is to find the maximum output values of each of the eight, hidden black box functions.

This is tricky because students begin with a very small dataset and can make only one additional query per function each week. For example, if they begin with 10 initial input and output data points for a function, after 5 weeks they will still have only 15 input and output datapoints for that function.

To add to the challenge, the form of each function is unknown at the outset and there is a delay of several days between the student submitting their queries and receiving the results.


## Why Bayesian Optimisation ?

There are two main drivers for using Bayesian Optimisation via a Gaussian Process for the capstone challenge:
1.	We have a very small dataset
2.	The cost of obtaining the result of new sample points is high. i.e. it takes a week to obtain the result of a new sample point.

Prior research indicates that Bayesian Optimisation and a Gaussian Process are effective methods to approach an optimisation challenge given the above constraints.

The most relevant ideas and techniques from that research are:

- We can create a surrogate model to represent the function we’re trying to map based on the limited number of datapoints we have at the beginning of the challenge

- We can quantify the degree of uncertainty of the model at points in between our known datapoints.

- We can use a variety of different acquisition functions to select promising regions of the search space to sample next.

- We can update the model each week, feeding our latest findings back in to improve the accuracy of the model

Together, these factors make a compelling case for using Bayesian Optimisation and a Gaussian Process to tackle this capstone challenge.

## Technical Approach

I’ve used Bayesian Optimisation by implementing:

- Numpy

- GaussianProcessRegressor from Scikit-learn

- ConstantKernel from Scikit-learn

- WhiteKernel from Scikit-learn

- Matern Kernel from Scikit-learn

- Upper Confidence Bound (UCB) acquisition functions

- Expected Improvement (EI) acquisition functions

I've also used:

- ChatGPT and Claude LLMs during the later stages of the project to help select sample points close to existing maxima

- Principal Compenent Analysis (PCA) to identify those features which make a significant difference to the result

## Week By Week Strategy

Week 1: I wasn’t sure what to expect so used a conservative, exploratory approach for all functions, ConstantKernal with an Upper Control Bound (UCB) acquisition function and Kappa 20. This yielded an improvement in the results of F4, F5 and F8 which was very encouraging.

Week 2, I increased Kappa to 20 for all functions except F1 and F2 to encourage even more exploration. I manually selected sample points for F1 and F2 in the middle of unexplored regions but they didn’t yield an improvement. However, the results of my Gaussian Process (GP) predictions for F5, F7 and F8 all improved.

Week 3, I decided to try an Expected Improvement (EI) acquisition function for all functions to get a better understanding of how it performs relative to UCB. It only improved the result of F8.

Week 4: I stuck with my EI acquisition function to give it a fair shot at proving itself to be effective. This yielded another improvement for F8 but no improvement for the other functions.

Week 5: I could still see some large unexplored regions in F1 and F2 so overrode my model again and manual selected sample points in the middle of those regions. I used my EI acquisition function for the other functions and lowered Xi from 0.1 to 0.01 for F8 only. This yielded an improvement for F6 but no other function.

Week 6: I decided to give EI one last try and used the same hyperparameters as the previous week except for F1, for which I manually selected another sample point. I made no gains in any of the functions this week.

Week 7: As we entered the second half of the capstone project, I switched back to UCB and lowered Kappa of 0.1 for all functions to encourage exploitation. I also switched from ConstantKernel to Matern Kernel for F8 hoping it would exploit the promising region that I’d found during the first 4 weeks. This yielded my first ever improvement for F3 and further improvements in F5, F6 and F7. This was my best week since the start of the project.

Week 8: The course introduced LLMs so I asked ChapGPT to recommend sample points for functions 1 and 2. I retained the ConstantKernal, UCB and low Kappa configuration for functions 3, 4 and 6, switched to Matern Kernel for F5 hoping to encourage even more exploitation and added WhiteKernel for F8 in case it was prone to noise. This yielded my first improvement for F2 but no improvement for any of the other functions.

Week 9: Because of the improvement I’d achieved using an LLM the previous week, I decided to use ChatGPT for all functions this week. I prompted it to use Temperature = 0.1, Top-k = 3, Top-p = 3 to bias it towards exploitation. This yielded an improvement in F5 and F6 but none of the other functions.

Week 10: I used ChatGPT for all functions again this week. I tightened up the prompts, telling it to select points close to my current best result and again told it to use Temperature = 0.1, Top-k = 3, Top-p = 3 in the prediction model. This yielded an improvement in F5 but none of the other functions.

Week 11: This week I used ChatGPT to recommend hyperparameters for my Gaussian Process and acquisition function. I fed it the dataset for one function at a time and asked it “Which hyperparameters should I use for my Gaussian Process in order to find a sample point which yields the highest possible result.” Then I ran my GP with those settings and checked the result it forecast for each function. In many cases, the forecast was less than my current best so I discarded most of those recommendations and reverted to the hyperparameters from Week 8 and manually selected a sample point for F1. This yielded my first ever improvement in F1 as well as improvements in F5 and F7.

Week 12: I switched most functions to Matern Kernel with a mixture of UCB and EI acquisition functions. I also applied PCA to all of the functions and used the findings to manually tweak a few of the parameters for F4 and F5 prior to submission. I also asked Claude to recommend a sample point for F6 in the same manner I’d prompted ChatGPT in weeks 9 and 10. This yielded improvements in F3 but none of the other functions.

Week 13: For the final week I decided not to make any major changes and continued to use ConstantKernel with UCB for F1, Matern Kernel with UCB for F2, F3, F6 and F7 and Matern Kernel with EI for F4, F5 and F8. This yielded final round improvements in F4 and F8.

The hyperparameters used to make the GP predictions each week can be found in the **Capstone Week n All Functions.py** files in the /code directory.

## Alternatives Considered

Other libraries I could have used are:

- Logistic Regression or Support Vector Machine from scikit learn.

- Tensorflow and Keras

- PyTorch

I’ve used Numpy, a Gaussian Process and LLMs because they perform particularly well given our very small dataset and the high cost of obtaining additional results.

I didn't use:

- Logistic Regression because I think it will struggle with non-linear data

- SVM because it’s sensitive to noise and requires data to be properly scaled

- Tensorflow, Keras or Pytorch because I don’t think they’ll cope well with our small dataset


## Documents Referenced

I’ve used these references which were provided by the faculty:

- Kelta, Zoumana. ‘Mastering Bayesian optimisation in data science Links to an external site..’ Datacamp.

- Chennu, Srivas, Andrew Maher, Christian Pangerl, et al. ‘Rapid and scalable Bayesian AB testing Links to an external site.’. IEEE. July 27, 2023. 

And this reference based on personal research:

- https://colab.research.google.com/github/krasserm/bayesian-machine-learning/blob/master/bayesian_optimization.ipynb#scrollTo=iV-8w2jSwlKv




