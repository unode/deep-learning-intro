---
title: "Classification by a Neural Network using Keras"
teaching: 0
exercises: 0
questions:
- "What is a neural network?"
- "How do I compose a Neural Network using Keras?"
- "How do I train this network on a dataset"
- "How do I get insight into learning process"
- "How do I measure the performance of the network"

objectives:
- "Explain what a classification task is"
- "Use one-hot-encoding to prepare data for classification in Keras"
- "Describe a fully connected layer"
- "Implement a fully connected layer with Keras"
- "Use Keras to train a small fully connected network on prepared data"
- "Plot the loss curve of the training process"
---


## Introduction
In this first episode we will learn how to create and train a Neural Network using Keras to solve a simple classification task.

The goal of this episode is to quickly get your hands dirty in actually defining and training a neural network, without going into depth of how neural networks work on a technical or mathematical level.
We do want you to go through the most deep learning workflow that was covered in the introduction.
As a reminder below are the steps of the deep learning workflow:
1. Formulate / Outline the problem
2. Identify inputs and outputs
3. Prepare data
4. Choose a cost function and metrics
5. Choose a pretrained model or start building architecture from scratch
6. Train model
7. Tune hyperparameters
8. 'predict'

In this episode will focus on a minimal example for each of these steps, later episodes will build on this knowledge to go into greater depth for some or all of these steps.
Furthermore this episode assumes you already know what a neural network is and what a classification
task is. However, below is a very short summary of what these are as a reminder.

### Neural Network
A neural network is an artificial intelligence technique loosely based on the way biological
neural networks work.
A neural network consists of connected computational units called neurons.
Each neuron takes the sum of all its inputs, performs some, typically non-linear, calculation on them and produces one output.
This calculation is called the activation function.
The connections between neurons are called edges, these edges typically have a weight associated with
them.
This weight determines the 'strength' of the connection, these weights are adjusted during training.
In this way, the combination of neurons and edges describe a computational graph, an example can be
seen in the image below.
In most neural networks neurons are aggregated into layers.
Signals travel from the input layer to the output layer, possibly through one or more intermediate layers called hidden layers.


![An example neural network with ][neural-network]
[*Glosser.ca, CC BY-SA 3.0 <https://creativecommons.org/licenses/by-sa/3.0>, via Wikimedia Commons*](https://commons.wikimedia.org/wiki/File:Colored_neural_network.svg)

### Classification Task
In a classification task the goal is to determine to which instance of a category a certain input
sample belongs to.
In such a classification task we will 'show' each sample to the neural network and have it make
a guess.
In the beginning, when the neural network is still untrained, it will likely 'guess' the wrong category, therefore we will need to train it.
During training the neural network is repeatedly asked to classify samples and the training
algorithm will steer the network towards the correct answers by updating the weights of the
neural network.

> ## GPU usage
> For this lesson having a GPU (graphics card) available is not needed.
> We specifically use very small toy problems so that you do not need one.
> However, Keras will use your GPU automatically when it is available.
> Using a GPU becomes necessary when tackling larger datasets or complex problems which require
> more complex Neural Network.
{: .callout}
## Step 1. Formulate / Outline the problem: Penguin classification
In this episode we will be using the [penguin dataset](https://zenodo.org/record/3960218), this is a dataset that was published in 2020 by Allison Horst and contains data on three different species of the penguins.

We will use the penguin dataset to train a neural network which can classify which species a
penguin belongs to, based on their physical characteristics.
> ## Goal
> The goal is to predict a penguins' species using the attributes available in this dataset.
{: .objectives}

The `palmerpenguins` data contains size measurements for three penguin species observed on three islands in the Palmer Archipelago, Antarctica.
The physical attributes measured are flipper length, beak length, beak width, body mass, and sex.

![Illustration of the three species of penguins found in the Palmer Archipelago, Antarctica: Chinstrap, Gentoo and Adele][palmer-penguins]
*Artwork by @allison_horst*

![Illustration of the beak dimensions called culmen length and culmen depth in the dataset][penguin-beaks]
*Artwork by @allison_horst*

These data were collected from 2007 - 2009 by Dr. Kristen Gorman with the [Palmer Station Long Term Ecological Research Program](https://pal.lternet.edu/), part of the [US Long Term Ecological Research Network](https://lternet.edu/). The data were imported directly from the [Environmental Data Initiative](https://environmentaldatainitiative.org/) (EDI) Data Portal, and are available for use by CC0 license ("No Rights Reserved") in accordance with the [Palmer Station Data Policy](https://pal.lternet.edu/data/policies).

## 2. Identify inputs and outputs
> ## Prerequisite: Start Jupyter Notebook
> Start a jupyter notebook by issueing the following command on a command line. It is probably a
> good idea to create an empty directory for this course first.
>
> ~~~
> $ mkdir deep-learning-intro
> $ cd deep-learning-intro
> $ jupyter notebook
> ~~~
> {:.language-bash}
> Now create a new notebook and call it assignment1.ipynb.
{:.prereq}

To identify the inputs and outputs that we will use to design the neural network we need to familiarize
ourselves with the dataset. This step is sometimes also called data exploration.

We will start by importing the [Seaborn](https://seaborn.pydata.org/) library that will help us get the dataset and visualize it.
Seaborn is a powerful library with many visualizations. Keep in mind it requires the data to be in a
pandas dataframe, luckily the datasets available in seaborn are already in a pandas dataframe.

~~~
import seaborn as sns
~~~
{:.language-python}

We can load the penguin dataset using
~~~
penguins = sns.load_dataset('penguins')
~~~
{:.language-python}

This will give you a pandas dataframe which contains the penguin data.

> ## penguin dataset
>
> Use seaborn to load the dataset and inspect the mentioned attributes.
> 1. What are the different features called in the dataframe?
> 2. Are the target classes of the dataset stored as numbers or strings?
> 3. How many samples does this dataset have?
>
> > ## Solution
> > **1.** Using the pandas `describe` function you can see the names of the features and some statistics:
> > ~~~
> > penguins.describe()
> > ~~~
> > {:.language-python}
> >
> >
> > |       | bill_length_mm | bill_depth_mm | flipper_length_mm | body_mass_g |
> > |------:|---------------:|--------------:|------------------:|------------:|
> > | count |     342.000000 |    342.000000 |        342.000000 |  342.000000 |
> > |  mean |      43.921930 |     17.151170 |        200.915205 | 4201.754386 |
> > |   std |       5.459584 |      1.974793 |         14.061714 |  801.954536 |
> > |   min |      32.100000 |     13.100000 |        172.000000 | 2700.000000 |
> > |   25% |      39.225000 |     15.600000 |        190.000000 | 3550.000000 |
> > |   50% |      44.450000 |     17.300000 |        197.000000 | 4050.000000 |
> > |   75% |      48.500000 |     18.700000 |        213.000000 | 4750.000000 |
> > |   max |      59.600000 |     21.500000 |        231.000000 | 6300.000000 |
> > {:.output}
> >
> > **2.** We can get the unique values in the `species` column using the `unique` function of pandas.
> > It shows the target class is stored as a string and has 3 unique values. This type of column is
> > usually called a 'categorical' column.
> >
> > ~~~
> > penguins["species"].unique()
> > ~~~
> > {:.language-python}
> > ~~~
> > ['Adelie', 'Chinstrap', 'Gentoo']
> > ~~~
> > {:.output}
> >
> > **3.** Using `describe` function on the species column shows there are 344 samples
> > unique species
> > ~~~
> > penguins["species"].describe()
> > ~~~
> > {:.language-python}
> > ~~~
> > count        344
> > unique         3
> > top       Adelie
> > freq         152
> > Name: species, dtype: object
> > ~~~
> > {:.output}
> {:.solution}
{:.challenge}

### Visualization
Looking at numbers like this usually does not give a very good intuition about the data we are
working with, so let us create a visualization.
#### Pair Plot
One nice visualization for datasets with relatively few attributes is the Pair Plot.
This can be created using `sns.pairplot(...)`. It shows a scatterplot of each attribute plotted against each of the other attributes.
By using the `hue='class'` setting for the pairplot the graphs on the diagonal are layered kernel density estimate plots.

> ## Create the pairplot using Seaborn
>
> Use the seaborn pairplot function to create a pairplot
> * Is there any class that is easily distinguishable from the others?
> * Which combination of attributes shows the best separation?
>
> > ## Solution
> > ~~~
> > sns.pairplot(df, hue="class")
> > ~~~
> > {:.language-python}
> > ![Pair plot showing the separability of the three species of penguin][pairplot]
> >
> > The plots show that the green class, Gentoo is somewhat more easily distinguishable from the other two.
> > The other two seem to be most easily separated by a combination of bill length and bill
> > depth.
> {:.solution}
{:.challenge}

### Input and Output Selection
Now that we have familiarized ourselves with the dataset we can select the data attributes to use
as input for the neural network and the target that we want to predict.

In the rest of this episode we will use the `bill_length_mm`, `bill_depth_mm`, `flipper_length_mm`, `body_mass_g` attributes.
The target for the classification task will be the `species`.

Using the following code we can select these columns from the dataframe:
~~~
input_data = penguins.drop(columns=["species", 'island', 'sex'])
target = penguins['species']
~~~
{:.language-python}

> ## Data Exploration
> Exploring the data is an important step to familiarize yourself with the problem and to help you
> determine the relavent inputs and outputs.
{:.keypoints}
## 3. Prepare data
The input data and target data are not yet in a format that is suitable to use for training a neural network.
During the exploration phase you may have noticed that some rows in the dataset have missing (NaN)
values, leaving such values in the input data will ruin the training, so we need to deal with them.
There are many ways to deal with missing values, but for now we will just remove the offending rows.

Add a call to `dropna()` before the input_data definition that we used above, the cell should now
look like this:
~~~
valid_data = penguins.dropna()

input_data = valid_data.drop(columns=["species", 'island', 'sex'])
target = valid_data['species']
~~~
{:.language-python}

Second, the target data is also in a format that cannot be used to train.
To train a neural network we need to be able to calculate how "far away" the species predicted by the
neural network is from the true species.
When the target is a string category column as we have here it is very difficult to determine this "distance" or error.
Therefore we will transform this column into a more suitable format.
Again there are many ways to do this, however we will be using the 1-hot encoding.
This encoding creates multiple columns, as many as there are unique values, and
puts a 1 in the column with the corresponding correct class, and 0's in
the other columns.
For instance, for a penguin of the Adelie species the 1 hot encoding would be 1 0 0

Fortunately pandas is able to generate this encoding for us.
~~~
target = pd.get_dummies(valid_data['species'])
target.head() # print out the top 5 to see what it looks like.
~~~
{:.language-python}

> ## Keras for Neural Networks
> For this lesson we will be using [Keras](https://keras.io/) to define and train our neural network
> models.
> Keras is a machine learning framework with ease of use as one of its main features.
> It is part of the tensorflow python package and can be imported using `from tensorflow import keras`.
>
> Keras includes functions, classes and definitions to define deep learning models, cost functions and
> optimizers (optimizers are used to train a model).
{:.callout}

## 4. Choose a cost function and metrics
Having selected the target enoding, we need to select an appropriate loss
function that we will use during training.
This loss function tells the training algorithm how wrong, or how 'far away' from the true
value the predicted value is.

For the one hot encoding we will use the Categorical Crossentropy loss.
This is a measure for how close the distribution of the three neural network outputs corresponds
to the distribution of the three values in the one hot encoding.
It is lower if the distributions are more similar.
In keras this is implemented in the `keras.losses.CategoricalCrossentropy` class.

## 5. Choose a pretrained model or start building architecture from scratch
Now we will build a neural network from scratch, and although this sounds like
a daunting task, with Keras it is actually surprisingly straightforward.

With keras you compose a neural network by creating layers and linking them
together. For now we will only use one type of layer called a fully connected
or Dense layer. In keras this is defined by the `keras.layers.Dense` class.

A dense layer has a number of neurons, which is a parameter when you create the layer.
When connecting the layer to its input and output layers every neuron in the dense
layer gets an edge (i.e. connection) to ***all*** of the input neurons and ***all*** of the output neurons.
The hidden layer in the image in the introduction of this episode is a Dense layer.

The input in Keras also gets special treatment, Keras autmatically calculates the number of inputs
and outputs a layer needs and therefore how many edges need to be created.
This means we need to let Keras now how big our input is going to be.
We do this by instantiating a `keras.Input` class and tell it how big our input is.

~~~
inputs = keras.Input(shape=input_data.shape[1])
~~~
{:.language-python}

We store a reference to this input class in a variable so we can pass it to the creation of
our hidden layer.
Creating the hidden layer can then be done as follows:
~~~
hidden_layer = keras.layers.Dense(10, activation="relu")(inputs)
~~~
{:.language-python}

The instantiation here has 2 parameters and a seemlingly strange combination of parenthenses, so
let's take a closer look.
The first parameter `10` is the number of neurons we want in this layer, this is one of the
hyperparameters of our system and needs to be chosen carefully. We will get back to this in the section
on hyperparameter tuning.
The second parameter is the activation function to use, here we choose relu which is 0 for values
0 and below and the same as the input for values above 0.
This is a commonly used activation functions in deep neural networks that is proven to work well.
Next we see an extra set of parenthenses with inputs in them, this means that after creating an
instance of the Dense layer we call it as if it was a function.
This tells the Dense layer to connect the layer passed as a parameter, in this case the inputs.
Finally we store a reference so we can pass it to the output layer in a minute.

Now we create another layer that will be our output layer.
Again we use a Dense layer and so the call is very similar to the previous one.
~~~
output_layer = keras.layers.Dense(3, activation="softmax")(hidden_layer)
~~~
{:.language-python}
Because we chose the one hot encoding, we use `3` neurons for the output layer.
This works well in combination with the `softmax` activation function and the
Categorical Crossentropy loss we chose earlier.

The softmax activation ensures that the three output neurons produce values in the range
(0, 1) and the sum to 1.
We can interpret this as a kind of 'probability' that the sample belongs to a certain
species.
The Categorical Crossentropy loss then works well with comparing these probabilities
with 'true' probabilities that we generated using the one hot encoding.

Now that we have defined the layers of our neural network we can combine them into
a keras model which facilitates training the network.
~~~
model = keras.Model(inputs=inputs, outputs=output_layer)
model.summary()
~~~
{:.language-python}

The model summary here can show you some information about the neural network we have defined.

> ## Create the neural network
>
> Using the code snippets above define a keras model with 1 hidden layer with
> 10 neurons and an output layer with 3 neurons.
>
> * How many parameters does the resulting model have?
> * What happens to the number of parameters if we increase or decrease the number of neurons
>   in the hidden layer?
>
> > ## Solution
> > ~~~
> > inputs = keras.Input(shape=input_data.shape[1])
> > hidden_layer = keras.layers.Dense(10, activation="relu")(inputs)
> > output_layer = keras.layers.Dense(3, activation="softmax")(hidden_layer)
> >
> > model = keras.Model(inputs=inputs, outputs=output_layer)
> > model.summary()
> > ~~~
> > {:.language-python}
> >
> > ~~~
> Model: "functional_1"
> > _________________________________________________________________
> > Layer (type)                 Output Shape              Param #
> > =================================================================
> > input_1 (InputLayer)         [(None, 4)]               0
> > _________________________________________________________________
> > dense (Dense)                (None, 10)                50
> > _________________________________________________________________
> > dense_1 (Dense)              (None, 3)                 33
> > =================================================================
> > Total params: 83
> > Trainable params: 83
> > Non-trainable params: 0
> > _________________________________________________________________
> > ~~~
> > {:.output}
> >
> > The model has 83 trainable parameters.
> > If you increase the number of neurons in the hidden layer the number of
> > trainable parameters increases and decreases if you decrease the number
> > of neurons.
> {:.solution}
{:.challenge}

## 6. Train model
We are now ready to train the model.
## 7. Tune hyperparameters

## 8. 'predict'


{% include links.md %}

[neural-network]: https://upload.wikimedia.org/wikipedia/commons/4/46/Colored_neural_network.svg "Neural Network"
{: width="25%"}

[palmer-penguins]: ../fig/palmer_penguins.png "Palmer Penguins"
{: width="50%"}

[penguin-beaks]: ../fig/culmen_depth.png "Culmen Depth"
{: width="50%"}

[pairplot]: ../fig/pairplot.png "Pair Plot"
{: width="66%"}