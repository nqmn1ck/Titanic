import time
from joblib import Parallel, delayed

from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import nqmDecisionTree
from randomForests import randomForests

def run_for_num_levels(num_levels, y_index, num_trees, max_feature_count, features, 
                       X_train, y_train, X_test, y_test):
    '''
    '''
    print(f"parallelizing for max_feature_count {max_feature_count}, num levels {num_levels}")
    # Entropy calculations
    tree = randomForests(
        y_index, num_trees, max_feature_count,
        features=features, X_train=X_train,
        y_train=y_train, X_test=X_test,
        y_test=y_test,
        num_levels=num_levels, gain='entropy'
    )
    tree.create_trees()
    predictions = tree.predict(X_train)
    entropy_train_accuracy = nqmDecisionTree.evaluate_predictions(predictions, np.array(y_train))
    predictions = tree.predict(X_test)  
    entropy_test_accuracy = nqmDecisionTree.evaluate_predictions(predictions, np.array(y_test))

    # Gini calculations
    tree = randomForests(
        y_index, num_trees, max_feature_count,
        features=features, X_train=X_train,
        y_train=y_train, X_test=X_test,
        y_test=y_test,
        num_levels=num_levels, gain='gini'
    )
    tree.create_trees()
    predictions = tree.predict(X_train)
    gini_train_accuracy = nqmDecisionTree.evaluate_predictions(predictions, np.array(y_train))
    predictions = tree.predict(X_test)
    gini_test_accuracy = nqmDecisionTree.evaluate_predictions(predictions, np.array(y_test))

    # Return the accuracies (in the same order as you were appending them)
    return (entropy_train_accuracy * 100,
            entropy_test_accuracy * 100,
            gini_train_accuracy * 100,
            gini_test_accuracy * 100)

def evaluate_forest(X_train, X_test, y_train, y_test, features):
    '''
    Hyperparameter search for number of features to consider to split on, and number of levels in the tree.
    Parallelized by running the random forest for each tree max level parameter simultaneously.
    TODO: search for the best number of trees.
    '''
    num_trees = 700
    max_features = len(features)

    # Collect accuracies across all possible max levels and for gini and entropy
    entropy_train_accuracies = []
    entropy_test_accuracies = []
    gini_train_accuracies = []
    gini_test_accuracies = []

    for max_feature_count in [1, 3, 6, 9, 11]:
        start = time.time()
        print(f"Creating and testing max_feature_count {max_feature_count}")

        # Parallel execution over num_levels
        results = Parallel(n_jobs=-1)(
            delayed(run_for_num_levels)(
                num_levels, 1, num_trees, max_feature_count, features, 
                X_train, y_train, X_test, y_test
            )
            for num_levels in [0, 3, 6, 9, 12]
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

    plot_forests_accuracies(
        [entropy_train_accuracies, entropy_test_accuracies],
        [gini_train_accuracies, gini_test_accuracies],
        max_features
    )

def plot_forests_accuracies(entropy_accuracies, gini_accuracies, max_feature_count):
    print("Creating Plots")
    # Accuracies are a 2D matrix, where the row is the number of max features from 0 to max_feature_count
    levels = [0, 3, 6, 9, 12]
    fig = plt.figure()
    axs = [[], []] # First row = entropy, second row = gini

    types = ['Entropy', 'Gini']

    # Initialize subplot positions. One row for each type, and one column for each max_feature_count selection
    feature_counts = [1, 3, 6, 9, 11]
    for i in range(len(feature_counts)):
        for type in range(len(types)):
            axs[type].append(fig.add_subplot(len(types), len(feature_counts), len(feature_counts) * type + i + 1)) # When type=entropy (0), adds to the first row. Second row when type=gini (1)

    # Create subplots. One row for each type, and one column for each max_feature_count selection
    feature_counts = [1, 3, 6, 9, 11]
    for i, type in enumerate(types):
        for j, feature_count in enumerate(feature_counts):
            if i == 0: 
                accuracies = entropy_accuracies
            else:
                accuracies = gini_accuracies

            axs[i][j].plot(levels, accuracies[0][j], label="train", marker='o')
            axs[i][j].plot(levels, accuracies[1][j], label="test", marker='o')
            if i == 0:
                axs[i][j].set_title(f"# of Sampled Features = {feature_count}\n\n\n")
            if i == 1:
                axs[i][j].set_xlabel("Number of levels", fontsize=13)
            if j == 0:
                axs[i][j].set_ylabel(f"Accuracy (%). Gain type = {type}", fontsize=13)
            axs[i][j].set_xticks(levels)
            axs[i][j].set_yticks(np.arange(0, 110, 10, dtype=np.int16))
            axs[i][j].set_yticklabels(np.arange(0, 110, 10, dtype=np.int16))
            axs[i][j].set_xticklabels(levels)
            axs[i][j].legend()

    fig.suptitle("Random Forests Hyperparameter Search for Accuracy on Test and Train")
    plt.savefig('plots/evaluate_forests.png')
    plt.show()

def evaluate_tree(X_train, X_test, y_train, y_test, features):

    # Collect accuracies across all possible max levels and for gini and entropy
    entropy_train_accuracies = []
    entropy_test_accuracies = []
    gini_train_accuracies = []
    gini_test_accuracies = []

    max_feature_count = len(features)
    max_feature_count = 2 

    for num_levels in range(0, max_feature_count + 1):
        print(f"Creating and testing Level {num_levels}")
        # Entropy
        y_index = 1
        tree = nqmDecisionTree(y_idx=y_index, features=features, X_train=X_train, num_levels=num_levels, gain='entropy')
        tree.create_tree(X_train)
        predictions = tree.predict(X_train)
        train_accuracy = nqmDecisionTree.evaluate_predictions(predictions, np.array(y_train))
        predictions = tree.predict(X_test)
        test_accuracy = nqmDecisionTree.evaluate_predictions(predictions, np.array(y_test))

        entropy_train_accuracies.append(train_accuracy * 100)
        entropy_test_accuracies.append(test_accuracy * 100)

        # Gini
        y_index = 1
        tree = nqmDecisionTree(y_idx=y_index, features=features, X_train=X_train, num_levels=num_levels, gain='gini')
        tree.create_tree(X_train)
        predictions = tree.predict(X_train)
        train_accuracy = nqmDecisionTree.evaluate_predictions(predictions, np.array(y_train))
        predictions = tree.predict(X_test)
        test_accuracy = nqmDecisionTree.evaluate_predictions(predictions, np.array(y_test))

        gini_train_accuracies.append(train_accuracy * 100)
        gini_test_accuracies.append(test_accuracy * 100)

    plot_tree_accuracies(
        [entropy_train_accuracies, entropy_test_accuracies],
        [gini_train_accuracies, gini_test_accuracies],
        max_feature_count
    )

def plot_tree_accuracies(entropy_accuracies, gini_accuracies, max_feature_count):
    levels = np.arange(0,max_feature_count + 1, 1)
    fig = plt.figure()
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

def main():
    train = pd.read_csv('data/train.csv').drop(columns=['Name', 'Ticket', 'Cabin']).dropna(
        subset=['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    )
    test = pd.read_csv('data/test.csv').drop(columns=['Name', 'PassengerId', 'Cabin', 'Ticket']).dropna(
        subset=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    )

    # Divide Age into categorical columns
    train['young_age'] = (train['Age'] <= 30).astype(int)
    train['medium_age'] = ((train['Age']) > 30 & (train['Age'] <= 60)).astype(int)
    train['old_age'] = (train['Age'] > 60).astype(int)
    train = train.drop(columns=['Age'])

    # Divide Fare into categorical columns
    train['average_fare'] = (train['Fare'] <= 52).astype(int)
    train['larger_fare'] = (train['Fare'] > 52).astype(int)
    train = train.drop(columns=['Fare'])

    features = train.columns.to_numpy()

    X = train.to_numpy()
    y = train['Survived'].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=69, test_size=0.25, shuffle=True)

    #evaluate_tree(X_train, X_test, y_train, y_test, features)
    #evaluate_forest(X_train, X_test, y_train, y_test, features)
    #evaluate_forest_treenum(X_train, X_test, y_train, y_test, features)

    y_index = 1

    # Default num trees = 1000, default num splits to considser = sqrt(num features), default gain = entropy
    tree = randomForests(
        X_train, features, y_index
    )

    tree.create_trees()

    predictions = tree.predict(X_train)
    train_accuracy = nqmDecisionTree.Utils.evaluate_predictions(predictions, np.array(y_train))

    predictions = tree.predict(X_test)
    test_accuracy = nqmDecisionTree.Utils.evaluate_predictions(predictions, np.array(y_test))

    print(f"Training accuracy: {train_accuracy}\nTest accuracy: {test_accuracy}")

if __name__ == "__main__":
    main()