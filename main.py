from sklearn.model_selection import train_test_split
import pandas as pd
from nqmDecisionTree import nqmDecisionTree
from randomForests import randomForests
import matplotlib.pyplot as plt
import math
import numpy as np

def evaluate_forest(X_train, X_test, y_train, y_test, features):
    '''
    Hyperparameter search for number of features to consider to split on, and number of levels in the tree
    TODO: search for the best number of trees.
    '''
    num_trees = 60
    max_features = len(features)

    # Collect accuracies across all possible max levels and for gini and entropy
    entropy_train_accuracies = []
    entropy_test_accuracies = []
    gini_train_accuracies = []
    gini_test_accuracies = []

    for max_feature_count in range(1, max_features):
        print(f"Creating and testing max_feature_count {max_feature_count}")
        max_feature_gini_train_accuracies = []
        max_feature_gini_test_accuracies = []
        max_feature_entropy_train_accuracies = []
        max_feature_entropy_test_accuracies = []

        for num_levels in range(0, max_features + 1):
            # Entropy
            y_index = 1
            tree = randomForests(
                y_index, num_trees, max_feature_count,
                features=features, X_train=X_train,
                y_train = y_train, X_test=X_test,
                y_test = y_test,
                num_levels=num_levels, gain='entropy'
            )
            tree.create_trees()
            predictions = tree.predict(X_train)
            train_accuracy = nqmDecisionTree.evaluate_predictions(predictions, np.array(y_train))
            predictions = tree.predict(X_test)
            test_accuracy = nqmDecisionTree.evaluate_predictions(predictions, np.array(y_test))

            max_feature_entropy_train_accuracies.append(train_accuracy * 100)
            max_feature_entropy_test_accuracies.append(test_accuracy * 100)

            # Gini
            y_index = 1
            tree = randomForests(
                y_index, num_trees, max_feature_count,
                features=features, X_train=X_train,
                y_train = y_train, X_test=X_test,
                y_test = y_test,
                num_levels=num_levels, gain='gini'
            )
            tree.create_trees()
            predictions = tree.predict(X_train)
            train_accuracy = nqmDecisionTree.evaluate_predictions(predictions, np.array(y_train))
            predictions = tree.predict(X_test)
            test_accuracy = nqmDecisionTree.evaluate_predictions(predictions, np.array(y_test))

            max_feature_gini_train_accuracies.append(train_accuracy * 100)
            max_feature_gini_test_accuracies.append(test_accuracy * 100)

        # Creates 2d matrix, where the row number = max feature count
        entropy_train_accuracies.append(max_feature_entropy_train_accuracies)
        entropy_test_accuracies.append(max_feature_entropy_test_accuracies)
        gini_train_accuracies.append(max_feature_gini_train_accuracies)
        gini_test_accuracies.append(max_feature_gini_test_accuracies)
        

    plot_forests_accuracies(
        [entropy_train_accuracies, entropy_test_accuracies],
        [gini_train_accuracies, gini_test_accuracies],
        max_features
    )

def evaluate_forest_treenum(X_train, X_test, y_train, y_test, features):
    '''
    Hyperparameter search for number of features for best number of trees
    '''

    #Potentiallly best options:
    #    Sampled features=1: i) entropy w/ num_levels = 6, gini w/ num_levels = 6
    #    Sampled features=2: i) entropy w/ numlevels = 5, gini w/ numlevels = 8

    for num_trees in range(1, 200, 10):
        print(f"\n**********\nNUM TREES: {num_trees}\n**********\n")
        # Collect accuracies across all possible max levels and for gini and entropy

        sampled_feature_counts = [1, 2]
        number_of_levels = [[6, 6], [5, 8]] 

        for i, sampled_feature_count in enumerate(sampled_feature_counts):
            levels = number_of_levels[i]
            y_index = 1
            tree = randomForests(
                y_index, num_trees, sampled_feature_count,
                features=features, X_train=X_train,
                y_train = y_train, X_test=X_test,
                y_test = y_test,
                num_levels=levels[0], gain='entropy'
            )
            tree.create_trees()
            predictions = tree.predict(X_train)
            train_accuracy = nqmDecisionTree.evaluate_predictions(predictions, np.array(y_train))
            predictions = tree.predict(X_test)
            test_accuracy = nqmDecisionTree.evaluate_predictions(predictions, np.array(y_test))
            print(f"Number of sampled features = {sampled_feature_count}\nMax level = {levels[0]}\nGain = 'entropy'\nTrain Accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}\n")

            # Gini
            tree = randomForests(
                y_index, num_trees, sampled_feature_count,
                features=features, X_train=X_train,
                y_train = y_train, X_test=X_test,
                y_test = y_test,
                num_levels=levels[1], gain='gini'
            )
            print(f"Number of sampled features = {sampled_feature_count}\nMax level = {levels[1]}\nGain = 'entropy'")
            tree.create_trees()
            predictions = tree.predict(X_train)
            train_accuracy = nqmDecisionTree.evaluate_predictions(predictions, np.array(y_train))
            predictions = tree.predict(X_test)
            test_accuracy = nqmDecisionTree.evaluate_predictions(predictions, np.array(y_test))
            print(f"Number of sampled features = {sampled_feature_count}\nMax level = {levels[1]}\nGain = 'entropy'\nTrain Accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}\n")


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

def plot_forests_accuracies(entropy_accuracies, gini_accuracies, max_feature_count):
    # Accuracies are a 2D matrix, where the row is the number of max features from 0 to max_feature_count
    levels = np.arange(0,max_feature_count + 1, 1)
    fig = plt.figure()
    axs = [[], []] # First row = entropy, second row = gini

    types = ['Entropy', 'Gini']

    for i in range(max_feature_count):
        for type in range(len(types)):
            axs[type].append(fig.add_subplot(nrows=len(types), ncols=len(max_feature_count), index=len(max_feature_count) * type + i))

    for i, type in enumerate(types):
        for j in range(max_feature_count):
            if i == 0: 
                accuracies = entropy_accuracies
            else:
                accuracies = gini_accuracies

            axs[i][j].plot(levels, accuracies[0], label="train", marker='o')
            axs[i][j].plot(levels, accuracies[1], label="test", marker='o')
            axs[i][j].set_title(f"{type} tree accuracy per number of levels")
            axs[i][j].set_xlabel("Number of levels")
            axs[i][j].set_ylabel("Accuracy (%)")
            axs[i][j].set_xticks(levels)
            axs[i][j].set_yticks(np.arange(0, 110, 10, dtype=np.int16))
            axs[i][j].set_yticklabels(np.arange(0, 110, 10, dtype=np.int16))
            axs[i][j].set_xticklabels(levels)
            axs[i][j].legend()

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
    num_trees = 700
    num_levels = 13
    num_split_features_to_consider = 10

    tree = randomForests(
        y_index, num_trees, m=num_split_features_to_consider,
        features=features, X_train=X_train,
        y_train = y_train, X_test=X_test,
        y_test = y_test,
        num_levels=13, gain='gini'
    )
    tree.create_trees()
    predictions = tree.predict(X_train)
    train_accuracy = nqmDecisionTree.evaluate_predictions(predictions, np.array(y_train))
    predictions = tree.predict(X_test)
    test_accuracy = nqmDecisionTree.evaluate_predictions(predictions, np.array(y_test))
    print(f"Train Accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}\n")

if __name__ == "__main__":
    main()