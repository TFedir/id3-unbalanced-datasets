# Hubert P

from math import log2
import numpy as np

from arr_utils import *


class DecisionTree:
    def __init__(self):
        self.root_node = None
        self.unique_vals_for_feature = []

    def fit_id3(self, data, labels):
        """Uses id3 algorithm to fit a decision tree to the data"""
        self.unique_vals_for_feature = [
            list(set([row[feature_idx] for row in data]))
            for feature_idx in range(len(data[0]))
        ]
        self.root_node = self.id3(data, labels)

    def predict(self, datapoint):
        datapoint = np.array(datapoint)
        current_node = self.root_node
        while not current_node.is_leaf():
            feature_value = datapoint[current_node.feature_idx]
            current_node = current_node.branches[feature_value]

        return current_node.label

    def id3(self, data, labels, feature_idxs=None):
        data = np.array(data)
        node = Node()
        # if not defined, consider all features
        if feature_idxs == None:
            feature_idxs = list(range(0, len(data[0])))

        unique_labels = np.unique(labels)

        # If all samples have the same class label, create a leaf node.
        if len(unique_labels) == 1:
            node.label = unique_labels[0]
            return node

        # If there are no features left, create a leaf node with the majority class label.
        if len(feature_idxs) == 0:
            node.label = get_most_occuring(labels)
            return node

        best_feature_idx = find_best_feature_idx(
            data, labels, feature_idxs=feature_idxs
        )
        node.feature_idx = best_feature_idx

        best_feature_values = self.unique_vals_for_feature[best_feature_idx]
        feature_splits = {
            value: list(
                filter(lambda x: x[0][best_feature_idx] == value, zip(data, labels))
            )
            for value in best_feature_values
        }

        # If no splits were found for the best feature, create a leaf node with the majority class label.
        if not feature_splits:
            node.label = get_most_occuring(labels)
            return node

        for feature_value, subset in feature_splits.items():
            # If no elements left in the subset, label with parent's majority class
            if not subset:
                node.label = get_most_occuring(labels)
                return node

            remaining_features = list(feature_idxs)  # deep copy

            # removes feature chosen in this iteration from list of features considered in latter iterations
            remaining_features.pop(remaining_features.index(best_feature_idx))

            # unzips the zip
            subset_data, subset_labels = zip(*subset)
            subset_data = list(subset_data)
            subset_labels = list(subset_labels)

            child_node = self.id3(
                data=subset_data, labels=subset_labels, feature_idxs=remaining_features
            )
            node.branches[feature_value] = child_node

        return node

    def print(self):
        print("-- ROOT --", end="")
        self.root_node.print()


class Node:
    def __init__(self):
        self.feature_idx = -1
        self.branches = {}
        self.label = None

    def is_leaf(self):
        return self.label != None

    def print(self, depth=0):
        TAB_DIFF = 4
        tabs = " " * depth
        print(
            f"""
{tabs}-- NODE --
{tabs}Feature idx: {self.feature_idx}
{tabs}Label: {self.label}
{tabs}Branch count: {len(self.branches)}"""
        )

        for branch in self.branches.items():
            print(f"{tabs}{branch[0]}: ", end="")
            branch[1].print(depth + TAB_DIFF)


def calculate_label_entropy(labels) -> float:
    """Calculates entropy of the dataset based on its labels"""
    num_samples = len(labels)
    probabilities = map(lambda x: x / num_samples, get_element_counts(labels).values())

    return sum([-p * log2(p) for p in probabilities])


def find_best_feature_idx(data, labels, feature_idxs) -> int:
    """Returns index of feature providing the highest information gain (lowest entropy)"""

    num_samples = len(labels)
    entropy_per_feature = {}

    for idx in feature_idxs:
        feature_values = np.unique([datapoint[idx] for datapoint in data])
        feature_avg_entropy = 0

        for value in feature_values:
            labeled_data = list(zip(data, labels))
            filtered_labels = [
                item[1] for item in labeled_data if item[0][idx] == value
            ]

            entropy = calculate_label_entropy(filtered_labels)
            feature_avg_entropy += entropy * len(filtered_labels) / num_samples

        entropy_per_feature[idx] = feature_avg_entropy

    return min(entropy_per_feature.items(), key=lambda x: x[1])[0]


# ---- Example usage ----
if __name__ == "__main__":
    data = [["A", "C"], ["A", "D"], ["B", "C"], ["B", "D"]]
    labels = ["True", "False", "False", "True"]

    tree = DecisionTree()
    tree.fit_id3(data, labels)

    tree.print()

    to_pred = ["B", "D"]
    print(f"Prediction for {to_pred}: {tree.predict(to_pred)}")
