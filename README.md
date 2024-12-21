# Titanic

## Goal:
Custom impementation of decision trees, k nearest neighbors, and neural nets to do ensembling and fully optimize our prediction.

## Current implementation:
Currently random forests & decision trees is the only fully implemented model. Running main.py without modification will use the default randomForests class parameters (num_trees = 1000, gain = 'entropy', num_levels = no limit, num features considered on each split = sqrt(len(features))).

To test the random forest and decision tree models, I have created 3 unittest files which use the Titanic dataset:
1) `test_decision_tree.py`: Tests the decision tree over all possible maximum depth lengths over each type of split algorithm (gini impurity or information gain) and plots the resulting train and test accuracy.
2) `test_random_forests_single.py`: Tests a single random forests instance with whatever parameters you choose, and prints the test and train accuracy to console.
3) `test_random_forests_parameters.py`: Parameter search on the parameter `num_trees` and parameter lists `max_feature_counts`, and `max_num_levels`. Uses multiprocessing to complete faster and plots the resulting test and train accuracy for every parameter combination.

If you have the dependencies installed, you can run any of these by simply going to the parent folder and running 

```python test/test_*.py```. Every test file also has easy-access parameter variables if you would like to tweak them yourself.
## Dependencies:
The only dependencies required are `sklearn, pandas, matplotlib, numpy`, and `python >= 3.12` Install with pip or conda.

## Features of each model:
### DecisionTree:
1) Allows fine tuning of parameters, including selection of entropy or gini and the max depth of the tree.
2) Handles new data with unseen patterns during prediction by probabilistically choosing a child node to continue down instead of throwing errors or dumping into an "other" branch.
### RandomForests:
1) Allows fine tuning of parameters "num_trees, "max levels/depth per tree", "max number of features to sample & consider at each split", and "gain type (gini or information)". 
