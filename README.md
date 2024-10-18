HomeKit Weather Station
=======================

This is an exploration project, to discover the possibilities of machine learning 
on a tiny microcontroller. The goal is to create a weather station that can predict
the weather based on the data it collects. 

Final device should have a Apple HomeKit integration no enable home automation and
voice control.

> **Note:** This project is in an active development phase, with intention to be 
> used as a learning dairy for myself.

## Working with the garage door up

This project is built "with the garage door up" mindset, inspired by Andy Matuschak's 
[notes](https://notes.andymatuschak.org/About_these_notes?stackedNotes=z21cgR9K3UcQ5a7yPsj2RUim3oM2TzdBByZu). 

I'm inviting you to see the work before it's finished, so you can follow the progress as it emerges.

This is "anti-marketing" because marketing is about promoting a product in the best possible light, 
whereas working with the garage door up exposes unpolished work and it is more realistic.

Feel free to reach me out with any questions or suggestions, or open an issue.

## Project log

### Model training

- [X] Download dataset from WorldWeatherOnline
    - Sample of 10 years 
    - Hourly frequency
    - Temperature, humidity, raining
- [X] Cleaning up the dataset
	- Convert rain data into a binary (Yes/No)
	- Balance the dataset by undersampling the majority class
	- Scale the input features with Z-score
- [X] Training the model with TensorFlow 
	- Split the dataset into train, validation, and test datasets
	- Create a model with Keras API
	- Analyze the accuracy and loss after each training epoch
- [X] Evaluating the model's effectiveness 
	- Visualize a confusion matrix
	- Calculate Recall, Precision, and F-score performance metrics
- [X] Quantizing
    - Convert the model to TFLite
    - On device deployment
- [ ] Code refactoring
    - Group functions into classes
    - Separate business logic of the app from the interface
    - Build a pipeline for continuous training

### On-device deployment

- [ ] Collecting data with DHT22 sensor
- [ ] On-device inference

## Prepare dataset

In order to make training the model easier and reproducible, I wrote a script
to download the dataset from Open Meteo through their API. 

Script supports multiple parameters, like location and date range.
To make it more useful, I added a feature to save the data to different formats,
like CSV, JSON, and column text, primarily used for debugging.

The script is pretty straightforward, I decided not to use asynchronous approach,
since script is going to be run only in one thread and only single request, it 
should be fast enough.

For both, user input and API response validation, I used `pydantic` library, with
some custom validators, primarily for processing multiple values from the user.

To make a prediction for the rain, we need to convert data that comes as a float,
into a binary value. I decided to use a simple threshold, if the value is greater
than 0.1, it's raining, otherwise it's not.

The temperature and humidity features have a different numerical ranges, and so 
different contributions during training, leading to a bias. I need to rescale using 
`z-score` technique to ensure that each input feature contributes equally during training.

## Data cleaning and preparation

Although the data we have is already quite clean, we still need to do some
post-processing to make it more suitable for training the model. For instance, 
rain data is represented as a float, but we need to convert it into a binary
value.

I decided to use a simple threshold, if the value is greater than 0.1, it's
raining, otherwise it's not. Maybe it makes sense to think about more cases,
like drizzle, but for now, I'll keep it simple.

Since the dataset is unbalanced, I decided to undersample the majority class (no rain 
in my case). Without this step, the model will fail to learn the patterns of the minority,
and performance of `recall` and `precision` metrics will be very poor.

I used the basic `z-score` technique to scale the temperature and humidity features,
to ensure that each input feature contributes equally during training. To validate the
results, 

Example of usage (including default parameters) can be found in the `Makefile`:

```bash
make dataset
```

## Training the model

I splitted the dataset into train, validation, and test subsets. I used a binary classification
model with one fully connected layer with 12 neurons and followed by ReLU activation function,
one dropout layer with 0.2 rate, and the output layer with a single neuron and sigmoid activation
function.

Use `make train` to train the model on downloaded dataset. 

## Evaluating the model

I calculated the confusion matrix, and common performance metrics: 

- `accuracy`: the ratio of correctly predictions to the total number of tests
- `recall`: metric tells us how many of the actual positive cases we were able to predict (higher is better)
- `precision`: how many of the predicted positive cases were actually positive (higher is better)
- `f1-score`: helps to evaluate recall and precision metrics at the same time (higher is better)

## Quantizing

Once the model is trained and evaluated, I need to compress it to allow inference on tiny devices. 
I exported model to keras format, and then used TFLite to convert it to FlatBuffer, applying
8-bit quantization to reduce the size. Final binary was converted to c-byte array to be used on
esp32 microcontroller.

For hex dump, I used `xxd` command, available on MacOS. For Linux you can install it with `sudo apt install xxd`.
See the `Makefile` for more details on which options to use.

## Reminders to myself

- Python packages, specially so complicated as TensorFlow, can be a nightmare to install. I spent
hours trying to reproduce Jupyter notebook environment both remote and local, and each time I ran
into a different issue. Best start for a new ML project is to create a robust environment with
Docker, and python dependencies, pinned to the specific version.
- I can't find any benefit of wrapping code into a cli tools, and fitting it into data pipeline
framework might be much more practical.
- During the work on the last part of model training, I noticed an interesting relation between the first step 
(dataset preparation), and next steps (training, quantizing). I should consider converting each step
into a separate class, and use a basic OOP principles to pass shared entities between steps, reducing
dependencies on global variables or non-trivial function executions between logically separated parts of the program.

