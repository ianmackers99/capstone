# LLMs
I used LLMs to recommend sample points and hyperparameters for some functions during weeks 8, 9, 10, 11 and 12

This directory contains the prompts I used and the results returned.

As can be seen from the **Week By Week** summary of my strategy in the main Readme for this repo, the LLM recommendations were not always sensible so I didn't always use them. This directory contains a record of the prompts and the raw results obtained, regardless of whether or not they were submitted for processing.

## This is how I used LLM recommendations

**Week 8:** The course introduced LLMs so I asked ChapGPT to recommend sample points for functions 1 and 2. This yielded my first improvement for F2 but no improvement for any of the other functions.

**Week 9:** Because of the improvement I’d achieved using an LLM the previous week, I decided to use ChatGPT for all functions this week. I prompted it to use Temperature = 0.1, Top-k = 3, Top-p = 3 to bias it towards exploitation. This yielded an improvement in F5 and F6 but none of the other functions.

**Week 10:** I used ChatGPT for all functions again this week. I tightened up the prompts, telling it to select points close to my current best result and again told it to use Temperature = 0.1, Top-k = 3, Top-p = 3 in the prediction model. This yielded an improvement in F5 but none of the other functions.

**Week 11:** This week I used ChatGPT to recommend hyperparameters for my Gaussian Process and acquisition function. I fed it the dataset for one function at a time and asked it “Which hyperparameters should I use for my Gaussian Process in order to find a sample point which yields the highest possible result.” Then I ran my GP with those settings and checked the result it forecast for each function. In many cases, the forecast was less than my current best so I discarded most of those recommendations and reverted to the hyperparameters from Week 8 and manually selected a sample point for F1. This yielded my first ever improvement in F1 as well as improvements in F5 and F7.

**Week 12:** I asked Claude to recommend a sample point for F6 in the same manner I’d prompted ChatGPT in weeks 9 and 10. This yielded no improvement in the results.
