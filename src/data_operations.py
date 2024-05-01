# Fedir T

from pandas import read_csv
import statistics
import numpy as np

german_map = {
    "A11": 1,
    "A12": 2,
    "A13": 3,
    "A14": 4,
    "A30": 1,
    "A31": 2,
    "A32": 3,
    "A33": 4,
    "A34": 5,
    "A40": 1,
    "A41": 2,
    "A42": 3,
    "A43": 4,
    "A44": 5,
    "A45": 6,
    "A46": 7,
    "A47": 8,
    "A48": 9,
    "A49": 10,
    "A410": 11,
    "A61": 1,
    "A62": 2,
    "A63": 3,
    "A64": 4,
    "A65": 5,
    "A71": 1,
    "A72": 2,
    "A73": 3,
    "A74": 4,
    "A75": 5,
    "A91": 1,
    "A92": 2,
    "A93": 3,
    "A94": 4,
    "A95": 5,
    "A101": 1,
    "A102": 2,
    "A103": 3,
    "A121": 1,
    "A122": 2,
    "A123": 3,
    "A124": 4,
    "A141": 1,
    "A142": 2,
    "A143": 3,
    "A151": 1,
    "A152": 2,
    "A153": 3,
    "A171": 1,
    "A172": 2,
    "A173": 3,
    "A174": 4,
    "A191": 1,
    "A192": 2,
    "A201": 1,
    "A202": 2,
}


def map_values_for_german(x):
    new_x = []
    for row in x:
        new_row = []
        for attr in row:
            if type(attr) == str:
                new_row.append(german_map[attr])
        new_x.append(new_row)
    return np.array(new_x)


def load_diabetes():
    dataframe = read_csv("./datasets/pima-indians-diabetes.csv", header=None)
    return dataframe.values


def load_haberman():
    dataframe = read_csv("./datasets/haberman.csv", header=None)
    return dataframe.values


def load_german():
    dataframe = read_csv("./datasets/german.csv", header=None)
    return dataframe.values


def discretize_data(data):
    columns = []
    columns = [[row[i] for row in data] for i in range(len(data[0]))]
    discretized_columns = []
    for column in columns:
        if type(column[0]) == str:
            discretized_columns.append(column)
            continue
        discretized_column = []
        mean = statistics.mean(column)
        for value in column:
            if value >= mean:
                discretized_column.append(2)
            else:
                discretized_column.append(1)
        discretized_columns.append(discretized_column)
    discretized_data = [
        [col[i] for col in discretized_columns] for i in range(len(data))
    ]
    return discretized_data
