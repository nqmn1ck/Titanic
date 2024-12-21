# Titanic

## Goal:
Custom impementation of decision trees, k nearest neighbors, and neural nets to do ensembling and fully optimize our prediction.

## Current implementation:
Currently random forests & decision trees is the only fully implemented model. Running main.py without modification will use the default randomForests class parameters (num_trees = 1000, gain = 'entropy', num_levels = no limit, num features considered on each split = sqrt(len(features))).

Hyperparameter search can be done by tweaking the evaluate_tree and evaluate_forest functions inside of main to search over different parameter values.

## Dependencies:
The only dependencies required are `sklearn, pandas, matplotlib, numpy`, and `python >= 3.12` Install with pip or conda.

## Features of each model:
### DecisionTree:
1) Allows fine tuning of parameters, including selection of entropy or gini and the max depth of the tree.
2) Handles new data with unseen patterns during prediction by probabilistically choosing a child node to continue down instead of throwing errors or dumping into an "other" branch.
