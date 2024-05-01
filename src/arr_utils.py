# Hubert P

import numpy as np


def get_element_counts(arr):
    unique = np.unique(arr)
    counts = {item: arr.count(item) for item in unique}

    return counts


def get_most_occuring(arr):
    """Returns the most occuring item in a collection"""
    counts = get_element_counts(arr)
    max_item = max(counts.items(), key=lambda x: x[1])

    return max_item[0]
