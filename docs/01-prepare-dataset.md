Prepare dataset
===============

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

For the temperature and humidity features, I need to rescale using `z-score` technique to 
insure that each input feature contributes equally during training.
