HomeKit Weather Station
=======================

This is an exploration project, to discover the possibilities of machine learning 
on a tiny microcontroller. The goal is to create a weather station that can predict
the weather based on the data it collects. 

Final device should have a Apple HomeKit integration no enable home automation and
voice control.

> **Note:** This project is in an active development phase, with intention to be 
> used as a learning dairy for myself.

## Project log

- [X] Download dataset from WorldWeatherOnline
    - Sample of 10 years 
    - Hourly frequency
    - Temperature, humidity, raining
- [X] Cleaning up the dataset
	- Convert rain data into a binary (Yes/No)
	- Balance the dataset by undersampling the majority class
	- Scale the input features with Z-score
- [ ] Training the model with TensorFlow 
	- Split the dataset into train, validation, and test datasets
	- Create a model with Keras API
	- Analyze the accuracy and loss after each training epoch
- [ ] Evaluating the model's effectiveness 
	- Visualize a confusion matrix
	- Calculate Recall, Precision, and F-score performance metrics
- [ ] Quantizing the model with TFLite converter
- [ ] Collecting data with DHT22 sensor
- [ ] On-device inference with TFLu
	- Circuit buffer 

## 0x01 Prepare dataset

In order to make training the model easier and reproducible, I wrote a script
to download the dataset from Open Meteo through their API. 

Script supports multiple different parameters, like location and date range.
To make it more useful, I added a feature to save the data to different formats,
like CSV, JSON, and column text, primarily used for debugging.

The script is pretty straightforward, I decided not to use asyncronous approach,
since script is going to be run only in one thread and only single request, it 
should be fast enough.

For both user input and API response validation I used `pydantic` library, with
some custom validators, primarily for processing multiple values from the user.

To make a prediction for the rain, we need to convert data that comes as a float,
into a binary value. I decided to use a simple threshold, if the value is greater
than 0.1, it's raining, otherwise it's not.

The temperature and humidity features have a different numerical ranges, and so 
different contributions during training, leading to a bias. I need to rescale using 
`z-score` technique to ensure that each input feature contributes equally during training.

## 0x02 Data cleaning and preparation

Although the data we have is already quite clean, we still need to do some
post-processing to make it more suitable for training the model. For instance, 
rain data is represented as a float, but we need to convert it into a binary
value.

I decided to use a simple threshold, if the value is greater than 0.1, it's
raining, otherwise it's not. Maybe it makes sense to think about more cases,
like drizzle, but for now, I'll keep it simple.

I used the basic `z-score` technique to scale the temperature and humidity features,
to ensure that each input feature contributes equally during training. To validate the
results, I plotted the histograms of the original and scaled data that can be found in
`tools/zscore_visualization.py` file. Forward `tools/dataset.py` script through the
pipe, example:

```sh
tools/dataset.py --start_date 2014-01-01 --end_date 2024-01-01 --location amsterdam \
                --format csv | tools/explore.py
```
## 0x03 Training the model

I splitted the dataset into train, validation, and test subsets. I used a binary classification
model with one fully connected layer with 12 neurons and followed by ReLU activation function,
one dropout layer with 0.2 rate, and the output layer with a single neuron and sigmoid activation
function.
