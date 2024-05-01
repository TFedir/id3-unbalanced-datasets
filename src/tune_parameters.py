# Fedir T

from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from test_model import discretize_data, load_diabetes, load_haberman, load_german
from sklearn.model_selection import KFold
from id3 import DecisionTree
from sklearn.metrics import f1_score
from statistics import mean
from data_operations import map_values_for_german


def oversample(x_train, y_train, strategy):
    ros = RandomOverSampler(random_state=50, sampling_strategy=strategy)
    x_over, y_over = ros.fit_resample(x_train, y_train)
    return (x_over, y_over)


def undersample(x_train, y_train, strategy):
    rus = RandomUnderSampler(random_state=50, sampling_strategy=strategy)
    x_under, y_under = rus.fit_resample(x_train, y_train)
    return (x_under, y_under)


def smote(x_train, y_train, strategy):
    over = SMOTE(random_state=50, sampling_strategy=strategy)
    x_over, y_over = over.fit_resample(x_train, y_train)
    return (x_over, y_over)


def main():
    strategies = range(60, 105, 5)
    strategies = [s / 100 for s in strategies]
    values = load_diabetes()
    X, Y = values[:, :-1], values[:, -1]
    X = discretize_data(X)
    # X = map_values_for_german(X)
    kfold = KFold(n_splits=5, shuffle=True)
    func_names = ["oversample", "undersample", "smote"]
    for i, func in enumerate([oversample, undersample, smote]):
        f_measures = []
        f_measures_for_strat = []
        for s in strategies:
            for train_idx, test_idx in kfold.split(X):
                x_train, y_train = [X[i] for i in train_idx], [Y[i] for i in train_idx]
                x_test, y_test = [X[i] for i in test_idx], [Y[i] for i in test_idx]
                x_over, y_over = func(x_train, y_train, s)
                tree = DecisionTree()
                tree.fit_id3(x_over, y_over)
                y_pred = []
                for x in x_test:
                    prediction = tree.predict(x)
                    y_pred.append(prediction)
                score = f1_score(y_test, y_pred, average="binary")
                f_measures_for_strat.append(score)
            f_measures.append(mean(f_measures_for_strat))
        f1_max = max(f_measures)
        best_strategy_idx = f_measures.index(f1_max)
        print(
            f"Best strategy for {func_names[i]}: {strategies[best_strategy_idx]}. F1 max: {f1_max}"
        )


if __name__ == "__main__":
    main()
