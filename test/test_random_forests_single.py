import unittest
import numpy as np
from base_test import baseTestTitanic

import os, sys
script_dir = os.path.dirname(os.path.abspath(__file__))
utils_dir = os.path.join(script_dir, '..', 'src')
sys.path.insert(0, utils_dir)
import models.tree.nqmDecisionTree as nqmDecisionTree
from models.tree.randomForests import randomForests

class testRandomForestsSingle(baseTestTitanic):
    def test_single(self):
        y_index = 1

        # Default num trees = 1000, default num splits to considser = sqrt(num features), default gain = entropy
        num_trees = 600
        num_levels = 5

        tree = randomForests(
            self.X_train, self.features, y_index, num_trees=num_trees, num_levels=num_levels
        )

        tree.create_trees()

        predictions = tree.predict(self.X_train)
        train_accuracy = nqmDecisionTree.Utils.evaluate_predictions(predictions, np.array(self.y_train))

        predictions = tree.predict(self.X_test)
        test_accuracy = nqmDecisionTree.Utils.evaluate_predictions(predictions, np.array(self.y_test))

        print(f"Training accuracy: {train_accuracy}\nTest accuracy: {test_accuracy}")

if __name__ == "__main__":
    unittest.main()