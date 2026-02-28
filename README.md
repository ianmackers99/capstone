# capstone
-- Project Overview --

The Black Box Optimisation (BBO) Capstone Project is a challenge designed to test a student’s knowledge and understanding of Machine Learning (ML) tools.

The overall goal of the capstone project is to find the maximum output values of 8 hidden black box functions. To achieve this, student’s use Bayesian Optimisation by creating a Gaussian Process surrogate of each function and an acquisition function to determine the next best place to sample the black box function. The results are fed back into the model each week so it progressively learns the shape of the functions.

This mimics a common real-world scenario where data scientists are asked to make predictions based on limited initial information and then update those predictions as additional data is received. Practical skills in this area are in great demand in the job market.


-- Input and Outputs --

At the beginning of the challenge, students are given a small, initial input and output dataset for each function in the form of Numpy arrays. The number of features/dimensions (2-8) and the number of initial samples (10-40) varies according to the function.

For a 2-dimensional dataset, i.e. with 2 features, the inputs might look like this:

•	0.66579958 0.12396913

•	0.87779099 0.7786275 

•	0.14269907 0.34900513

And the corresponding outputs might look like this:

•	0.53899612

•	0.42058624

•	-0.06562362

Students use these X and Y values to create a Gaussian Process to act as a surrogate for the true function.

Students then use an acquisition function to generate a new sample point for each of the 8 functions. The sample points must be presented as a sequence of hyphen-separated numbers in the range 0 and 1, rounded to 6 decimal places. Each number represents the value of a feature and together the numbers form a query which the black box will consume and then generate a result. The number of features varies by function so a typical query for a function with two features would look like this:

0.237256-0.049198

An eight-feature function would have eight hyphen-separated numbers, etc.

The results returned by the black box might look like this:

•	-1.06562362

Students add the new queries and corresponding results to the Numpy arrays each week and use them to train their model the following week.


-- Challenge Objectives --

The overall goal of the capstone project is to find the maximum output values of each of the eight, hidden black box functions.

This is tricky because students begin with a very small dataset and can only make one additional query per function per week. So, if they begin with 10 initial input and output data points for a function, after 5 weeks they will still have only 15 input and output datapoints for that function.

To add to the challenge, the form of each function is unknown at the outset and there is a delay of several days between the student submitting their queries and receiving the results.


-- Why Bayesian Optimisation ? --

There are two main drivers for using Bayesian Optimisation via a Gaussian Process for the capstone challenge:
1.	We have a very small dataset
2.	The cost of obtaining the result of new sample points is high. i.e. it takes a week to obtain the result of a new sample point.

Prior research indicates that Bayesian Optimisation and a Gaussian Process are effective methods to approach an optimisation challenge given the above constraints.

The most relevant ideas and techniques from that research are:

•	We can create a surrogate model to represent the function we’re trying to map based on the limited number of datapoints we have at the beginning of the challenge

•	We can quantify the degree of uncertainty of the model at points in between our known datapoints.

•	We can use a variety of different acquisition functions to select promising regions of the search space to sample next.

•	We can update the model each week, feeding back our latest findings an improving the accuracy of the model

Together, these factors make a compelling case for using Bayesian Optimisation and a Gaussian Process to tackle the capstone challenge.

-- Technical Approach --

I’ve used Bayesian Optimisation by implementing the Scikit Learn Gaussian Process library with a ConstantKernel and either an Upper Control Bound (UCB) or Expected Improvement (EI) acquisition function. I’ve considered other complementary methods such as logical regression and Support Vector Machines but don’t feel the benefit they might offer merits the additional complexity or investigation at present.

For week 1, I wasn’t sure what to expect so used a conservative, exploratory approach for all functions

For week 2, I realised my model wasn’t exploring sufficiently. So, I increased the Kappa value of my UCB acquisition function for all target functions except F1 and F2. I overrode my model and manually selected sample points for F1 and F2 after visualising the plot. Neither of the results I chose manually yielded an improvement but several of my other queries did.

For week 3, I decided to conduct an experiment and use an Expected Improvement acquisition function for all target functions to get a better understanding of how it performs relative to UCB. It made gains in three out of eight functions. 

-- Alternatives Considered --

At present, the main libraries I’m using for the BBO challenge are:

•	Numpy

•	GaussianProcessRegressor from Scikit-learn

•	ConstantKernel from Scikit-learn

Other libraries I could have used are:

•	Logistic Regression or Support Vector Machine from scikit learn.

•	Tensorflow and Keras

•	PyTorch

I’ve used Numpy, a Gaussian Process and ConstantKernel because they perform particularly well given our very small dataset and the high cost of obtaining additional results.

I haven’t used:
•	Logistic Regression because I think it will struggle with non-linear data

•	SVM because it’s sensitive to noise and requires data to be properly scaled

•	Tensorflow, Keras or Pytorch because I don’t think they’ll cope well with our small dataset



-- Documents Referenced --

I’ve used these resources which were provided by the faculty:

•	Kelta, Zoumana. ‘Masting Bayesian optimisation in data science Links to an external site..’ Datacamp.

•	Chennu, Srivas, Andrew Maher, Christian Pangerl, et al. ‘Rapid and scalable Bayesian AB testing Links to an external site.’. IEEE. July 27, 2023. 

And this resource based on personal research:

•	https://colab.research.google.com/github/krasserm/bayesian-machine-learning/blob/master/bayesian_optimization.ipynb#scrollTo=iV-8w2jSwlKv




