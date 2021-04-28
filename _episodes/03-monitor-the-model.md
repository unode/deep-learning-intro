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

### Select a subset and split into data (X) and labels (y)
The full dataset comprises 10 years (3654 days) from which we here will only select the first 3 years.
In addition, we will remove all columns of the place that we want to make predictions on (here: Düsseldorf which is about in the middle of all 18 locations).

~~~
columns_selected = [x for x in data.columns if not x.startswith("DUSSELDORF") if x not in ["DATE", "MONTH"]]
X_data = data.loc[:365*3][columns_selected]
X_data.head()
~~~
{:.language-python}

As a label, that is the the values we want to later predict, we here pick the sunshine hours which we can get by
~~~
y_data = data.loc[:365*3]["DUSSELDORF_sunshine"].values
~~~
{:.language-python}

### Split data and labels into training, validation, and test set
As with classical machine learning techniques, it is common in deep learning to split off a *test set* which remains untouched during model training and tuning. It is then later be used to evaluate the model performance. Here, we will also split off an additional *validation set*, the reason of which will hopefully become clearer later in this lesson.

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

~~~
def create_nn(n_features, n_predictions):
    # Input layer
    input = Input(shape=(n_features,), name='input')

    # Dense layers
    layers_dense = Dense(100, 'relu')(input)
    layers_dense = Dense(50, 'relu')(layers_dense)

    # Output layer
    output = Dense(n_predictions)(layers_dense)

    return Model(inputs=input, outputs=output)

n_features = X_data.shape[1]
n_predictions = 1
model = create_nn(n_features, n_predictions)
model.compile(loss='mse', optimizer=Adam(), metrics=['mse', 'mae'])
model.summary()
~~~
{:.language-python}

## Train a dense neural network
