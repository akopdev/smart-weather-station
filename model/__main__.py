from datetime import datetime

import pandas as pd
import sklearn.metrics
import tensorflow as tf
from callisto import Callisto
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers

from .data import DataProvider

app = Callisto()


# ~~~
# Prepare data
# ~~~


@app.task(name="source_data")
def get_data(location: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """Fetch weather data from provider."""
    provider = DataProvider(location)
    return provider.get_data(start_date, end_date)


@app.task(name="data")
def add_rain_column(source_data: pd.DataFrame) -> pd.DataFrame:
    source_data["rain"] = source_data["rain"].apply(lambda x: 1 if x > 0 else 0)
    return source_data


@app.task(name="balanced_data")
def undersampling_majority_class(data: pd.DataFrame) -> pd.DataFrame:
    """Undersample majority class"""

    df_majority: pd.DataFrame = data[data["rain"] == 0]
    df_minority: pd.DataFrame = data[data["rain"] == 1]

    # Undersample the majority class to match the minority class size
    df_majority_undersampled = df_majority.sample(len(df_minority), random_state=42)

    # Combine the minority class with the undersampled majority class
    df_balanced = pd.concat([df_minority, df_majority_undersampled])

    # Shuffle the resulting balanced dataset
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    return df_balanced


@app.task(name="zscore_data")
def calculate_zscore(balanced_data: pd.DataFrame) -> pd.DataFrame:
    """Calculate z-score for temperature and humidity."""
    for col in ["temperature_2m", "relative_humidity_2m"]:
        balanced_data[f"{col}_zscore"] = (
            balanced_data[col] - balanced_data[col].mean()
        ) / balanced_data[col].std(ddof=0)
    return balanced_data


# ~~~
# Train model
# ~~~


@app.task
def split_data(zscore_data: pd.DataFrame, train_size: float, validate_size: float):

    x = zscore_data[["temperature_2m_zscore", "relative_humidity_2m_zscore"]]
    y = zscore_data["rain"]

    # Split the data into train and test sets with 85% and 15%
    x_train, x_validate_test, y_train, y_validate_test = train_test_split(
        x, y, test_size=train_size, random_state=1
    )

    # Split the remaining data into validation and test sets
    x_test, x_validate, y_test, y_validate = train_test_split(
        x_validate_test, y_validate_test, test_size=validate_size, random_state=3
    )
    return x_train, x_validate, x_test, y_train, y_validate, y_test


@app.task(name="model")
def train_model(split_data, epochs: int, batch_size: int):
    # Define the model
    model = tf.keras.Sequential()
    model.add(layers.Dense(12, activation="relu", input_shape=(2,)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1, activation="sigmoid"))
    model.summary()
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    t_train, t_validate, t_test, h_train, h_validate, h_test = split_data

    model.fit(
        t_train,
        h_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        validation_data=(t_validate, h_validate),
    )
    return model


@app.task
def evaluate_and_export(split_data, model: tf.keras.Model, model_storage_name: str):
    t_train, t_validate, t_test, h_train, h_validate, h_test = split_data

    y_test_pred = model.predict(t_test)

    y_test_pred = (y_test_pred > 0.5).astype("int32")

    cm = sklearn.metrics.confusion_matrix(h_test, y_test_pred)

    TN = cm[0][0]
    TP = cm[1][1]
    FN = cm[1][0]
    FP = cm[0][1]

    accuracy = (TP + TN) / (TP + TN + FN + FP)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f_score = (2 * recall * precision) / (recall + precision)

    print("Training evaluation")
    print("====================")
    print("Confusion matrix: ", cm)
    print("Accuracy:         ", round(accuracy, 3))
    print("Recall:           ", round(recall, 3))
    print("Precision:        ", round(precision, 3))
    print("F-score:          ", round(f_score, 3))

    model.export(model_storage_name)


# ~~~
# Model quantization
# ~~~


@app.task
def quantization(model_storage_name: str, split_data):
    t_train, t_validate, t_test, h_train, h_validate, h_test = split_data

    def representative_dataset():
        for i_value in tf.data.Dataset.from_tensor_slices(t_test).batch(1).take(100):
            i_value_f32 = tf.dtypes.cast(i_value, tf.float32)
            yield [i_value_f32]

    converter = tf.lite.TFLiteConverter.from_saved_model(model_storage_name)
    converter.representative_dataset = representative_dataset
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()

    with open(f"{model_storage_name}.tfile", "wb") as f:
        f.write(tflite_model)


if __name__ == "__main__":
    app.run(
        location="Amsterdam",
        start_date=datetime(2000, 1, 1),
        end_date=datetime(2021, 1, 1),
        epochs=20,
        batch_size=64,
        train_size=0.15,
        validate_size=0.50,
        model_storage_name="var/rain_forecast_model",
    )
