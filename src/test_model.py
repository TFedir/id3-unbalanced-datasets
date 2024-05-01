# Fedir T

from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from id3 import DecisionTree
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from statistics import mean
from data_operations import (
    load_diabetes,
    load_german,
    load_haberman,
    discretize_data,
    map_values_for_german,
)


def main():
    values = load_german()
    X, Y = values[:, :-1], values[:, -1]
    X = discretize_data(X)
    X = map_values_for_german(X)
    kfold = KFold(n_splits=5, shuffle=True)
    f_measures = []
    func_names = [
        "no technique",
        "oversampling",
        "undersampling",
        "SMOTE",
        "SMOTE+undersampling",
    ]
    for i, func in enumerate(
        [no_sampling, oversample, undersample, smote, smote_and_under]
    ):
        for train_idx, test_idx in kfold.split(X):
            x_train, y_train = [X[i] for i in train_idx], [Y[i] for i in train_idx]
            x_test, y_test = [X[i] for i in test_idx], [Y[i] for i in test_idx]
            x_over, y_over = func(x_train, y_train)
            tree = DecisionTree()
            tree.fit_id3(x_over, y_over)
            y_pred = []
            for x in x_test:
                prediction = tree.predict(x)
                y_pred.append(prediction)
            score = f1_score(y_test, y_pred, average="binary")
            f_measures.append(score)

        print(f"{func_names[i]} F-measure: {mean(f_measures)}")


def oversample(x_train, y_train):
    ros = RandomOverSampler(random_state=50, sampling_strategy=0.7)
    x_over, y_over = ros.fit_resample(x_train, y_train)
    return (x_over, y_over)


def undersample(x_train, y_train):
    rus = RandomUnderSampler(random_state=50, sampling_strategy=0.65)
    x_under, y_under = rus.fit_resample(x_train, y_train)
    return (x_under, y_under)


def no_sampling(x_train, y_train):
    return (x_train, y_train)


def smote(x_train, y_train):
    over = SMOTE(random_state=50, sampling_strategy=0.65)
    x_over, y_over = over.fit_resample(x_train, y_train)
    return (x_over, y_over)


def smote_and_under(x_train, y_train):
    over = SMOTE(random_state=50, sampling_strategy=0.65)
    under = RandomUnderSampler(random_state=50, sampling_strategy=1)
    steps = [("o", over), ("u", under)]
    pipeline = Pipeline(steps=steps)
    x, y = pipeline.fit_resample(x_train, y_train)
    return (x, y)


if __name__ == "__main__":
    main()
