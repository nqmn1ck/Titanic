import unittest
from base_test import baseTestTitanic
import time
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import numpy as np

import os, sys
script_dir = os.path.dirname(os.path.abspath(__file__))
utils_dir = os.path.join(script_dir, '..', 'src')
sys.path.insert(0, utils_dir)
import models.tree.nqmDecisionTree as nqmDecisionTree
from models.tree.randomForests import randomForests

class testRandomForestsParameters(baseTestTitanic):

    def test_random_forests_parameters(self):
        '''
        Hyperparameter search for number of features to consider to split on, and number of levels in the tree.
        Parallelized by running the random forest for each tree max level parameter simultaneously.
        TODO: search for the best number of trees.
        '''
        #---------------------------------
        # TUNE THESE HYPERPARAMETER RANGES
        #---------------------------------

        # Number of trees per random forest
        num_trees = 5 

        # Max number of features to sample for consideration per split
        max_feature_counts = [1, 3, 6, 9, 11]

        # Max depth/level of each tree
        max_num_levels = [0, 3, 6, 9, 12]

        #---------------------------------

        # Collect accuracies across all possible max levels and for gini and entropy
        entropy_train_accuracies = []
        entropy_test_accuracies = []
        gini_train_accuracies = []
        gini_test_accuracies = []

        for max_feature_count in max_feature_counts:
            start = time.time()
            print(f"Creating and testing max_feature_count {max_feature_count}")

            # Parallel execution over num_levels
            results = Parallel(n_jobs=-1)(
                delayed(self.run_for_num_levels)(
                    num_levels, 1, num_trees, max_feature_count, self.features, 
                    self.X_train, self.y_train, self.X_test, self.y_test
                )
                for num_levels in max_num_levels
            )
            end = time.time()
            print(f"Finished in {end - start}")

            # results is a list of tuples: (entropy_train, entropy_test, gini_train, gini_test)
            # Unpack them into separate lists
            max_feature_entropy_train_accuracies = [r[0] for r in results]
            max_feature_entropy_test_accuracies  = [r[1] for r in results]
            max_feature_gini_train_accuracies    = [r[2] for r in results]
            max_feature_gini_test_accuracies     = [r[3] for r in results]

            # Append the results
            entropy_train_accuracies.append(max_feature_entropy_train_accuracies)
            entropy_test_accuracies.append(max_feature_entropy_test_accuracies)
            gini_train_accuracies.append(max_feature_gini_train_accuracies)
            gini_test_accuracies.append(max_feature_gini_test_accuracies)

        self.plot_forests_accuracies(
            [entropy_train_accuracies, entropy_test_accuracies],
            [gini_train_accuracies, gini_test_accuracies],
            max_feature_counts, max_num_levels
        )

    def run_for_num_levels(self, num_levels, y_index, num_trees, max_feature_count, features, 
                           X_train, y_train, X_test, y_test):
        print(f"parallelizing for max_feature_count {max_feature_count}, num levels {num_levels}")
        # Entropy calculations
        tree = randomForests(
            X_train, features, y_index, num_trees, max_feature_count, num_levels, gain='entropy'
        )
        tree.create_trees()
        predictions = tree.predict(X_train)
        entropy_train_accuracy = nqmDecisionTree.Utils.evaluate_predictions(predictions, np.array(y_train))
        predictions = tree.predict(X_test)  
        entropy_test_accuracy = nqmDecisionTree.Utils.evaluate_predictions(predictions, np.array(y_test))

        # Gini calculations
        tree = randomForests(
            X_train, features, y_index, num_trees, max_feature_count, num_levels, gain='gini'
        )
        tree.create_trees()
        predictions = tree.predict(X_train)
        gini_train_accuracy = nqmDecisionTree.Utils.evaluate_predictions(predictions, np.array(y_train))
        predictions = tree.predict(X_test)
        gini_test_accuracy = nqmDecisionTree.Utils.evaluate_predictions(predictions, np.array(y_test))

        # Return the accuracies (in the same order as you were appending them)
        return (entropy_train_accuracy * 100,
                entropy_test_accuracy * 100,
                gini_train_accuracy * 100,
                gini_test_accuracy * 100)


    def plot_forests_accuracies(self, entropy_accuracies, gini_accuracies, max_feature_counts, max_num_levels):
        print("Creating Plots")
        # Accuracies are a 2D matrix, where the row is the number of max features from 0 to max_feature_count
        fig = plt.figure(figsize=(15, 15))
        axs = [[], []] # First row = entropy, second row = gini

        types = ['Entropy', 'Gini']

        # Initialize subplot positions. One row for each type, and one column for each max_feature_count selection
        for i in range(len(max_feature_counts)):
            for type in range(len(types)):
                axs[type].append(fig.add_subplot(len(types), len(max_feature_counts), len(max_feature_counts) * type + i + 1)) # When type=entropy (0), adds to the first row. Second row when type=gini (1)

        # Create subplots. One row for each type, and one column for each max_feature_count selection
        for i, type in enumerate(types):
            for j, feature_count in enumerate(max_feature_counts):
                if i == 0: 
                    accuracies = entropy_accuracies
                else:
                    accuracies = gini_accuracies

                axs[i][j].plot(max_num_levels, accuracies[0][j], label="train", marker='o')
                axs[i][j].plot(max_num_levels, accuracies[1][j], label="test", marker='o')
                if i == 0:
                    axs[i][j].set_title(f"# of Sampled Features = {feature_count}\n\n\n")
                if i == 1:
                    axs[i][j].set_xlabel("Number of levels", fontsize=13)
                if j == 0:
                    axs[i][j].set_ylabel(f"Accuracy (%). Gain type = {type}", fontsize=13)
                axs[i][j].set_xticks(max_num_levels)
                axs[i][j].set_yticks(np.arange(0, 110, 10, dtype=np.int16))
                axs[i][j].set_yticklabels(np.arange(0, 110, 10, dtype=np.int16))
                axs[i][j].set_xticklabels(max_num_levels)
                axs[i][j].legend()

        fig.suptitle("Random Forests Hyperparameter Search for Accuracy on Test and Train")
        plt.savefig('plots/evaluate_forests.png')
        plt.show()

if __name__ == "__main__":
    unittest.main()