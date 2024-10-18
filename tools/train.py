#!/usr/bin/env python3

import argparse
import logging
import sys

import pandas as pd
import sklearn.metrics
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


def model_init():
    model = tf.keras.Sequential()
    model.add(layers.Dense(12, activation="relu", input_shape=(2,)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1, activation="sigmoid"))
    model.summary()

    # Compile the model
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


def evaluate(x_test: pd.DataFrame, y_test: pd.DataFrame, model: tf.keras.Model):
    y_test_pred = model.predict(x_test)

    y_test_pred = (y_test_pred > 0.5).astype("int32")

    cm = sklearn.metrics.confusion_matrix(y_test, y_test_pred)

    TN = cm[0][0]
    TP = cm[1][1]
    FN = cm[1][0]
    FP = cm[0][1]

    accuracy = (TP + TN) / (TP + TN + FN + FP)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f_score = (2 * recall * precision) / (recall + precision)

    # TODO: Instead of printing the evaluation, perform comparison and throw an error,
    #       if the model performs not good enough.
    print("Training evaluation")
    print("===================")
    print("Confusion matrix: ", cm)
    print("Accuracy:         ", round(accuracy, 3))
    print("Recall:           ", round(recall, 3))
    print("Precision:        ", round(precision, 3))
    print("F-score:          ", round(f_score, 3))
    return True


class Quantization:
    def __init__(self, path_to_model: str, test_data: pd.DataFrame):
        self.model = path_to_model
        self.test_data = test_data

    def get_representative_data(self):
        """Select a few hundred of samples randomly from the test dataset to calibrate the quantization"""
        for i_value in tf.data.Dataset.from_tensor_slices(self.test_data).batch(1).take(100):
            i_value_f32 = tf.dtypes.cast(i_value, tf.float32)
        yield [i_value_f32]

    def get_converter(self):
        converter = tf.lite.TFLiteConverter.from_saved_model(self.model)
        converter.representative_dataset = tf.lite.RepresentativeDataset(
            self.get_representative_data
        )
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        return converter

    def converter(self, filename: str) -> bool:
        converter = self.get_converter()
        tflite_model = converter.convert()
        try:
            with open(filename, "wb") as f:
                f.write(tflite_model)
        except Exception as e:
            log.error(e)
            return False
        return True


def train(data: pd.DataFrame, epochs: int = 20, batch_size: int = 64):
    t_train, t_validate, t_test, h_train, h_validate, h_test = split_data(
        data[["temperature_2m_zscore", "relative_humidity_2m_zscore"]], data["rain"]
    )
    model = model_init()
    model.fit(
        t_train,
        h_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        validation_data=(t_validate, h_validate),
    )

    if evaluate(t_test, h_test, model):
        # TODO: don't mix the model evaluation with the quantization
        filename = "var/rain_forecast_model"
        model.export(filename)
        quantize = Quantization(filename, t_test)
        if not quantize.converter("var/rain_forecast_model.tflite"):
            return
        return model


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("dataset", type=str, help="Path to the dataset")
    try:
        args = parser.parse_args()
        data = pd.read_csv(args.dataset)
    except Exception as e:
        sys.exit(e)

    model = train(data)
    if not model:
        sys.exit("Model could not be trained")


if __name__ == "__main__":
    main()
