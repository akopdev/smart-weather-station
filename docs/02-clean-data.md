Data cleaning and preparation
=============================

Although the data we have is already quite clean, we still need to do some
preprocessing to make it more suitable for training the model. For instance, 
rain data is represented as a float, but we need to convert it into a binary
value.

I decided to use a simple threshold, if the value is greater than 0.1, it's
raining, otherwise it's not. Maybe it make sense to think about more cases,
like drizzle, but for now, I'll keep it simple.
