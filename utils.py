import numpy as np


def convert_prob_to_label(data):
    return np.argmax(data, axis=-1)
