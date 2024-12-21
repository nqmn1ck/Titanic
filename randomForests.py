import nqmDecisionTree
import numpy as np

class randomForests():
    def __init__(self, X_train, features, y_idx, num_trees=1000, m=None,  num_levels=None, gain='entropy'):
        self.X_train = X_train
        self.features = features
        self.y_idx = y_idx
        self.num_trees = num_trees 
        self.m = m
        if self.m == None:
            self.m = int(np.ceil(np.sqrt(len((features)))))
        self.num_levels = num_levels
        if self.num_levels == None:
            self.num_levels = len(self.features)
        self.gain = gain

        self.trees = []
    
    def create_trees(self) -> None:
        for i in range(self.num_trees):
            # Bootstrap the training set
            X_index_choices = np.array([np.random.choice(np.arange(len(self.X_train)), 1)[0] for _ in range(len(self.X_train))])
            X_boot = self.X_train[X_index_choices,:]

            # Initialize the tree object
            decision_tree = nqmDecisionTree.DecisionTree(
                self.y_idx, self.features,
                X_boot, self.gain,
                num_levels=self.num_levels, random_forests=True,
                max_features=self.m
            )

            # Build the tree and append to our list of trees
            decision_tree.create_tree(X_boot)
            self.trees.append(decision_tree)
    
    def predict(self, X):
        trees = self.trees
        assert isinstance(trees, list[nqmDecisionTree.DecisionTree]), "Wrong tree object list is of wrong type"
        total_predictions = []
        for i, tree in enumerate(trees):
            predictions = tree.predict(X)
            total_predictions.append(predictions)

        # Sum along columns
        total_predictions = np.array(total_predictions).astype(np.int16)
        sums = np.sum(total_predictions, axis=0, dtype=int) #shape should be 1xn

        voted_predictions = []
        # Calculate prediction by majority vote of the trees
        for sum in sums:
            total_entries = total_predictions.shape[0]
            half = total_entries/2
            result = 0
            if sum >= half:
                result = 1
            voted_predictions.append(result)

        return np.array(voted_predictions)