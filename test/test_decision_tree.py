import unittest
import matplotlib.pyplot as plt
import numpy as np
from base_test import baseTestTitanic

import os, sys
script_dir = os.path.dirname(os.path.abspath(__file__))
utils_dir = os.path.join(script_dir, '..', 'src')
sys.path.insert(0, utils_dir)
from models.tree import nqmDecisionTree as nqmDecisionTree

class testDecisionTree(baseTestTitanic):
    def test_decision_tree(self):

        # Collect accuracies across all possible max levels and for gini and entropy
        entropy_train_accuracies = []
        entropy_test_accuracies = []
        gini_train_accuracies = []
        gini_test_accuracies = []

        max_feature_count = len(self.features)

        for num_levels in range(0, max_feature_count + 1):
            print(f"Creating and testing Level {num_levels}")
            # Entropy
            y_index = 1
            tree = nqmDecisionTree.DecisionTree(y_index, self.features, self.X_train, num_levels=num_levels, gain='entropy')
            tree.create_tree(self.X_train)
            predictions = tree.predict(self.X_train)
            train_accuracy = nqmDecisionTree.Utils.evaluate_predictions(predictions, np.array(self.y_train))
            predictions = tree.predict(self.X_test)
            test_accuracy = nqmDecisionTree.Utils.evaluate_predictions(predictions, np.array(self.y_test))

            entropy_train_accuracies.append(train_accuracy * 100)
            entropy_test_accuracies.append(test_accuracy * 100)

            # Gini
            y_index = 1
            tree = nqmDecisionTree.DecisionTree(y_index,self.features,self.X_train, num_levels=num_levels, gain='gini')
            tree.create_tree(self.X_train)
            predictions = tree.predict(self.X_train)
            train_accuracy = nqmDecisionTree.Utils.evaluate_predictions(predictions, np.array(self.y_train))
            predictions = tree.predict(self.X_test)
            test_accuracy = nqmDecisionTree.Utils.evaluate_predictions(predictions, np.array(self.y_test))

            gini_train_accuracies.append(train_accuracy * 100)
            gini_test_accuracies.append(test_accuracy * 100)

        self.plot_tree_accuracies(
            [entropy_train_accuracies, entropy_test_accuracies],
            [gini_train_accuracies, gini_test_accuracies],
            max_feature_count
        )

    def plot_tree_accuracies(self, entropy_accuracies, gini_accuracies, max_feature_count):
        levels = np.arange(0,max_feature_count + 1, 1)
        fig = plt.figure(figsize=(15, 15))
        axs = []
        axs.append(fig.add_subplot(121))
        axs.append(fig.add_subplot(122))

        type = ['Entropy', 'Gini']
        for i, type in enumerate(type):
            if i == 0: 
                accuracies = entropy_accuracies
            else:
                accuracies = gini_accuracies

            axs[i].plot(levels, accuracies[0], label="train", marker='o')
            axs[i].plot(levels, accuracies[1], label="test", marker='o')
            axs[i].set_title(f"{type} tree accuracy per number of levels")
            axs[i].set_xlabel("Number of levels")
            axs[i].set_ylabel("Accuracy (%)")
            axs[i].set_xticks(levels)
            axs[i].set_yticks(np.arange(0, 110, 10, dtype=np.int16))
            axs[i].set_yticklabels(np.arange(0, 110, 10, dtype=np.int16))
            axs[i].set_xticklabels(levels)
            axs[i].legend()

        plt.savefig('plots/evaluate_tree.png')
        plt.show()

if __name__ == "__main__":
    unittest.main()