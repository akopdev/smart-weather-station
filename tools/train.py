#!/usr/bin/env python3
import argparse
import logging
import sys

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import activations, layers
from typing import List

log = logging.getLogger(__name__)


def split_data(
    x: pd.DataFrame, y: pd.DataFrame, train_size: float = 0.15, validate_size: float = 0.50
):
    # Split the data into train and test sets with 85% and 15%
    x_train, x_validate_test, y_train, y_validate_test = train_test_split(
        x, y, test_size=train_size, random_state=1
    )

    # Split the remaining data into validation and test sets
    x_test, x_validate, y_test, y_validate = train_test_split(
        x_validate_test, y_validate_test, test_size=validate_size, random_state=3
    )
    return x_train, x_validate, x_test, y_train, y_validate, y_test


def model_init(data: List[str]):
    model = tf.keras.Sequential()
    model.add(layers.Dense(12, activation="relu", input_shape=(len(data),)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1, activation="sigmoid"))
    model.summary()
    return model


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("dataset", type=str, help="Path to the dataset")
    try:
        args = parser.parse_args()
        args.dataset
        data = pd.read_csv(args.dataset)
        t_train, t_validate, t_test, h_train, h_validate, h_test = split_data(
            data["temperature_2m"], data["relative_humidity_2m"]
        )
        print("Train data:")
        print(t_train, h_train)
    except Exception as e:
        sys.exit(e)


if __name__ == "__main__":
    main()
