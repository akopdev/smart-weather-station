HomeKit Weather Station
=======================

This is an exploration project, to discover the possibilities of machine learning 
on a tiny microcontroller. The goal is to create a weather station that can predict
the weather based on the data it collects. 

Final device should have a Apple HomeKit integration no enable home automation and
voice control.

## Project plan

- [X] [Download dataset from WorldWeatherOnline](docs/01-prepare-dataset.md)
    - Sample of 10 years 
    - Hourly frequency
    - Temperature, humidity, raining
- [ ] [Cleaning up the dataset](docs/02-clean-dataset.md)
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
