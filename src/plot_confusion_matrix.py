# Fedir T

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from id3 import DecisionTree
from data_operations import (
    load_diabetes,
    load_german,
    load_haberman,
    discretize_data,
    map_values_for_german,
)
from test_model import no_sampling, oversample, undersample, smote, smote_and_under


def plot_cmat(y_test, y_pred, func_name):
    cmat = confusion_matrix(y_test, y_pred)
    d = ConfusionMatrixDisplay(cmat)
    d.plot()
    plt.title("Confusion matrix for " + func_name)
    plt.savefig(f"./plots/german_{func_name}.png")
    plt.show()


def main():
    values = load_german()
    X, Y = values[:, :-1], values[:, -1]
    X = discretize_data(X)
    X = map_values_for_german(X)
    func_names = [
        "no technique",
        "oversampling",
        "undersampling",
        "SMOTE",
        "SMOTE+undersampling",
    ]
    x_train, x_test, y_train, y_test = train_test_split(X, Y)
    x_train = x_train.astype("int")
    y_train = y_train.astype("int")
    for i, func in enumerate(
        [no_sampling, oversample, undersample, smote, smote_and_under]
    ):
        x_over, y_over = func(x_train, y_train)
        tree = DecisionTree()
        tree.fit_id3(x_over, y_over)
        y_pred = []
        for x in x_test:
            prediction = tree.predict(x)
            y_pred.append(prediction)
        plot_cmat(y_test.tolist(), y_pred, func_names[i])


if __name__ == "__main__":
    main()
