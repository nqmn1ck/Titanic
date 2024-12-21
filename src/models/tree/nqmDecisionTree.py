import numpy as np

class DecisionTree():
    '''
    Basic Decision Tree Class.

    Features: Choose between gini or information gain. Choose maximum number of levels. 
    '''
    def __init__(self, y_idx, features: np.array, X_train: np.array, gain='entropy', alpha=1e-16, num_levels=None, random_forests = False, max_features=None):
        self.gain_type = gain
        self.y_idx = y_idx
        self.features = features
        self.alpha = alpha
        self.num_levels = num_levels
        if self.num_levels == None:
            self.num_levels = len(features) # default to j
        self.probabilities = self.predict_probabilities(X_train)
        self.random_forests = random_forests
        self.max_features = max_features

    def predict_probabilities(self, X: np.array) -> np.array:
        '''
        Predict probabilities of each feature value by
        computing their percentage of distribution in the X data
        '''
        num_samples = X.shape[0]
        probabilities = {}
        for i, feature in enumerate(self.features):
            feature_values = np.unique(X[:,i])
            value_probabilities = {}
            for value in feature_values:
                value_probabilities[value] = (len(X[:,i][X[:,i] == value]) + self.alpha) / num_samples
            probabilities[feature] = value_probabilities
        return probabilities

    def entropy(self, X: np.array, feature_idx: int) -> float:
        '''
        Calculate the entropy for a single feature
        '''
        entropy = 0
        for feature_value in np.unique(X[:,feature_idx]):
            probability = self.probabilities[self.features[feature_idx]][feature_value]
            entropy -= probability * np.log(probability)
        return entropy
    
    def gini(self, X: np.array, feature_idx: int) -> float:
        '''
        Calculate the Gini Impurity fo feature
        '''
        gini = 1
        for feature_value in np.unique(X[:,feature_idx]):
            probability = self.probabilities[self.features[feature_idx]][feature_value]
            gini -= probability ** 2 
        return gini


    def gain(self, X: np.array, child_split_idx: int, type: str) -> float:
        '''
        Given a node with any set of included features,
        compute the information gain for a certain feature split.
        '''
        parent = 0
        # Sum because we want the total entropy across all parent features
        for i in range(len(self.features)):
            if type == 'entropy':
                parent += self.entropy(X, i)
            elif type == 'gini':
                parent += self.gini(X, i)

        split_feature_values = np.unique(X[:,child_split_idx])
        child_proportional_sum = 0
        for feature_value in split_feature_values:
            # Create sample subset where the feature has the certain split value
            X_new = X[X[:,child_split_idx] == feature_value]
            proportion = len(X_new) / len(X)
            child = 0 
            for i in range(len(self.features)):
                if type == 'entropy':
                    child -= self.entropy(X_new, i)
                elif type == 'gini':
                    child -= self.gini(X_new, i)
            child_proportional_sum += proportion * child
        
        gain = parent - child_proportional_sum
        return gain
    
    def split_node(self, X: np.array, last_split_value: str, available_splits: np.array, level: int):
        '''
        Recursive algorithm that creates tree by splitting on the max gain, and returns dictionary representation.
        '''
        # If the level is 0, we have done all splits.
        if level == self.num_levels or len(available_splits) <= 0:
            # Choose if survived or died via max vote
            num_survived = len(X[X[:,self.y_idx] == 1])
            num_died = len(X[X[:,self.y_idx] == 0])

            if num_survived >= num_died:
                choice = "1"
            else:
                choice = "0"
            return choice

        # 1. Calculate information gain of each split, and choose the highest infogain split.
        # Uses randomforests random feature subselection if this is a randm forests instance
        max_gain = float("-inf")
        split_idx = -1
        if not self.random_forests or self.max_features == None:
            for i in available_splits:
                gain = self.gain(X, i, self.gain_type)
                if gain >= max_gain:
                    max_gain = gain
                    split_idx = i
        else:
            split_subselection = list(np.random.choice(available_splits, self.max_features))
            for i in split_subselection:
                gain = self.gain(X, i, self.gain_type)
                if gain >= max_gain:
                    max_gain = gain
                    split_idx = i
        
        # 2. Recursively split again on the split nodes until the level is zero
        available_splits_new = np.delete(available_splits, np.where(available_splits == split_idx))

        split_name = self.features[split_idx]
        tree = {}

        feature_values = np.unique(X[:,split_idx])
        for feature_value in feature_values:
            X_new = X[X[:,split_idx] == feature_value]
            tree[feature_value] = self.split_node(X_new, feature_value, available_splits_new, level+1)
        
        return {split_name: tree}
    
    def create_tree(self, X: np.array):
        available_splits = np.arange(len(self.features))
        available_splits = np.delete(available_splits, np.where(available_splits == self.y_idx)) # Don't split on the target
        self.tree = self.split_node(X, None, available_splits, 0) 
    
    def traverse_decision_tree(self, tree, sample):
        """
        Recursively traverses the decision tree based on the sample features.
        """
        try:
            for feature, value in tree.items():
                if isinstance(value, dict):
                    feature_idx = np.where(self.features == feature)
                    sample_feature_value = sample[feature_idx][0]
                    try:
                        next_node = value[sample_feature_value]
                    except KeyError:
                        # Unknown feature value encountered. Probabilistically sample a feature value from the available options.

                        # 1) Get the current options from the tree
                        possible_values = []
                        for item in list(value.items()):
                            possible_values.append(item[0])
                        possible_values = np.array(possible_values)

                        if len(possible_values) == 1:
                            # If only one option, do that option
                            next_node = value[possible_values[0]]
                        else:
                            # If multiple options, sample from the probability distribution of those options and choose
                            # continue down that branch
                            probabilities = []
                            for feature_value in possible_values:
                                probabilities.append(self.probabilities[feature][feature_value])
                            probabilities = np.array(probabilities)

                            # Normalize probabilities to sum to 1
                            probabilities_sum = sum(probabilities)
                            for i in range(len(probabilities)):
                                probabilities[i] = probabilities[i] / probabilities_sum

                            # Sample next feature
                            feature_value = np.random.choice(possible_values, 1, p=probabilities)[0]
                            next_node = value[feature_value]

                    if isinstance(next_node, dict):
                        return self.traverse_decision_tree(next_node, sample)
                    else:
                        return int(next_node)
                else:
                    # Base case if max level = 1
                    return value
        except AttributeError:
            # Happens when 
            return int(tree)

    def predict(self, X) -> np.array:
        # Initialize lists to store predictions and true labels
        correct = 0
        predictions = []
        for i, sample in enumerate(X):
            # Extract the true label
            # Traverse the decision tree to get the prediction
            prediction = self.traverse_decision_tree(self.tree, sample)
            predictions.append(prediction)

        return np.array(predictions)
    
class Utils():
    def evaluate_predictions(predictions: np.array, labels: np.array) -> int:
        '''
        Returns the accuracy (percentage) of the predictions
        '''
        total = predictions.shape[0]
        total_correct = sum(predictions == labels)
        accuracy = total_correct / total
        return accuracy
