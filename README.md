# Titanic
Custom random forests implementation to solve the titanic survival prediction problem. Running main.py without modification will use the default randomForests class parameters (num_trees = 1000, gain = 'entropy', num_levels = no limit, num features considered on each split = sqrt(len(features))).

Hyperparameter search can be done by tweaking the evaluate_tree and evaluate_forest functions inside of main to search over different parameter values.

# Dependencies:
The only dependencies required are sklearn, pandas, matplotlib, numpy, and python 3.12 or higher. Install with pip or conda.