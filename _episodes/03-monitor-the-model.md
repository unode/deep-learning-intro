---
title: "Monitor the training process"
teaching: 0
exercises: 0
questions:
- tbd
objectives:
- "Explain the importance of splitting the training data"
- "Use the data splits to plot the training process"
- "Measure the performance of your deep neural network"
- "Interpret the training plots to recognize overfitting"
- "Implement basic strategies to prevent overfitting"
- "Understand the effect of regularization techniques"

keypoints:
- "Separate training, validation, and test sets allows monitoring and evaluating your model."
- "Dropout is a way to prevent overfitting"
---

## Import dataset
~~~
filename_data = os.path.join(path_data, "weather_prediction_dataset.csv")
data = pd.read_csv(filename_data)
data.head()
~~~
{:.language-python}


~~~
columns_selected = [x for x in data.columns if not x.startswith("DUSSELDORF") if x not in ["DATE", "MONTH"]]
X_data = data.loc[:365*3][columns_selected]
X_data.head()
~~~


## Regression and classification - how to set a training goal
- Explain how to define the output part of a neural network
- What is the loss function (and which one to chose for a regression or classification task)?
In episode 2 we trained a dense neural network on a *classification task*. For this one hot encoding was used together with a Categorical Crossentropy loss function.
This measured how close the distribution of the neural network outputs corresponds to the distribution of the three values in the one hot encoding.
Now we want to work on a *regression task*, thus not prediciting the right class for a datapoint but a certain value (could in principle also be several values). In our example we want to predict the sunshine hours in Düsseldorf (or any other place in the dataset) for a particular day based on the weather data of all other places. 

### Network output layer:
The network should hence output a single float value which is why the last layer of our network will only consist of a single node. 

### Loss function:
The loss is what the neural network will be optimized on during training, so chosing a suitable loss function is crucial for training neural networks.
In the given case we want to stimulate that the prodicted values are as close as possible to the true values. This is commonly done by using the *mean squared error* (mse) or the *mean absolute error* (mae), both of which should work OK in this case. Often, mse is prefered over mae because it "punishes" large prediction errors more severely.
In keras this is implemented in the `keras.losses.MeanSquaredError` class.

## Train a dense neural network
