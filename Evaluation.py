import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def clear_false_index(*data_frames):
    """Clears the false indexing created by Pandas when saved as a CSV.
    WARNING:  ONLY USE ONCE PER DATA FRAME.

    :param data_frames: Data Frames
    :return: None
    """
    for df in data_frames:
        df.drop(df.columns[0], axis=1, inplace=True)


def make_sub_set(keys, data_frame):
    """Creates subsets of data by passing in a List of column keys and a Data Frame
    to pull from

    :param keys: List of String Keys
    :param data_frame: Data Frame
    :return: Data Frame
    """
    df = data_frame[keys]
    df.columns = keys
    return df


def simple_feature_scale(key, data_frame):
    """Uses Simple Feature Scaling to normalize a column.
    Col / Col.max()

    :param key: String Key
    :param data_frame: Data Frame
    :return: Data Frame Column
    """
    return data_frame[key]/data_frame[key].max()


def create_bins(key, data_frame, div):
    """Creates a Numpy Linspace based on the min/max of the column,
    and how many dividers you pass in.

    :param key: String Key
    :param data_frame: Data Frame
    :param div: Integer
    :return: Numpy Linspace
    """
    return np.linspace(min(data_frame[key]), max(data_frame[key]), div)


def create_binned_column(key, bin_names, data_frame):
    """Creates a binned column

    :param key: String Key
    :param bin_names: List of String Keys
    :param data_frame: Data Frame
    :return: Data Frame Column
    """
    return pd.cut(data_frame[key],
                  create_bins(key, data_frame, len(bin_names) + 1),
                  labels=bin_names, include_lowest=True)

