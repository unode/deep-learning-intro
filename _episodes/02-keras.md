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
{:.language-python}
{:.prereq}

To identify the inputs and outputs that we will use to design the neural network we need to familiarize
ourselves with the dataset. This step is sometimes also called data exploration.

We will start by importing the [Seaborn](https://seaborn.pydata.org/) library that will help us get the dataset and visualize it.
Seaborn is a powerful library with many visualizations. Keep in mind it requires the data to be in a
pandas dataframe, luckily the datasets available in seaborn are already in a pandas dataframe.

~~~
import seaborn as sns
~~~
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
Again there are many ways to do this, but two of the most used ones are
1. Mapping each value to a numerical value (e.g. Adelie => 0, Chinstrap => 1, Gentoo => 2),
   this is called Ordinal encoding.
2. Mapping the values using a 1-hot encoding. This encoding creates multiple columns, as many as there
   are unique values, and puts a 1 in the column with the corresponding correct class, and 0's in
   the other columns. (e.g. for the first row: 1 0 0)

We will try both encodings to train the neural network and see which one works best,
so let's create them both now.
Fortunately pandas is able to generate both encodings for us.
~~~
# Convert the target from string to category type
# so pandas can generate the codes for us.
target = target.astype('category')

target_numerical = target.cat.codes
target_1_hot = pd.get_dummies(target)
~~~
{:.language-python}

> ## Keras for Neural Networks
> For this lesson we will be using [Keras](https://keras.io/) to define and train our neural network
> models.
> Keras is a machine learning framework with ease of use as one of its main features.
> It is part of the tensorflow python package and can be imported using `from tensorflow import keras`.
>
> Keras includes functions, classes and definitions to define deep learning models, cost functions and
> optimizers (optimizer train a model).
{:.callout}

## 4. Choose a cost function and metrics
Having selected the target enodings that we want to try we need to select an appropriate loss
function that we will use during training.
This loss function tells the training algorithm how wrong, or how 'far away' from the true
value the predicted value is.

For the ordinal encoding the most straightforward loss function is the Mean Squared Error.
This loss function favours small deviations above large deviations.
In keras this is implemented in the `keras.losses.MeanSquaredError` class, which we will use later.

For the one hot encoding we will use the Categorical Crossentropy loss.
This is a measure for how close the distribution of the three neural network outputs corresponds
to the distribution of the three values in the one hot encoding.
Its lower if the distributions are more similar.

## 5. Choose a pretrained model or start building architecture from scratch

## 6. Train model

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