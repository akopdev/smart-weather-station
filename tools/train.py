#!/usr/bin/env python3
import argparse
import logging
import sys
from typing import List

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers

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

    # Compile the model
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


def train(data: pd.DataFrame, model_name: str):
    t_train, t_validate, t_test, h_train, h_validate, h_test = split_data(
        data["temperature_2m"], data["relative_humidity_2m"]
    )
    model = model_init(data)
    model.fit(
        t_train,
        h_train,
        epochs=100,
        batch_size=10,
        verbose=1,
        validation_data=(t_validate, h_validate),
    )
    model.save(model_name)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("dataset", type=str, help="Path to the dataset")
    try:
        args = parser.parse_args()
        data = pd.read_csv(args.dataset)
    except Exception as e:
        sys.exit(e)

    train(data, model_name="rain_forecast_model")


if __name__ == "__main__":
    main()
