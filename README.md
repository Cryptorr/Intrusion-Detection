# Intrusion Detection

Best accuracy: 0.9667.

The jupyter notebook shows the latest status.

Brief summary: We tried random forests and extremely random trees using manual fine-tuning (the best result stems from a random forest classifier). We then tried to use the auto-ML packages hyperopt [1, 2] and auto-sklearn [3, 4] to automate parts of the model search for us, but essentially had to capitulate as we're inexperienced with those packages as of now and ran into too many issues (see jupyter notebook sections 3. and 4.).

[1] https://hyperopt.github.io/hyperopt/
[2] https://hyperopt.github.io/hyperopt-sklearn/
[3] https://automl.github.io/auto-sklearn/stable/
[4] http://www.kdnuggets.com/2017/01/current-state-automated-machine-learning.html
