---
title: "Introduction to Keras"
teaching: 0
exercises: 0
questions:
- "How do I compose a Neural Network using Keras?"
- "How do I train this network on a dataset"
- "How do I get insight into learning process"
- "How do I measure the performance of the network"
- "What happens to the number of parameters if you add another layer?"

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
1. Formulate/ Outline the problem
2. Identify inputs and outputs
3. Prepare data
4. Choose a cost function and metrics
5. Choose a pretrained model or start building architecture from scratch
6. Train model
7. Tune hyperparameters
8. 'predict'

In this episode will focus on a minimal example for each of these steps, later episodes will build on this knowledge to go into greater depth for some or all of these steps.

> ## GPU usage
> For this lesson having a GPU (graphics card) available is not needed.
> We specifically use very small toy problems so that you do not need one.
> However, Keras will use your GPU automatically when it is available.
> Using a GPU becomes necessary when tackling larger datasets or complex problems which require
> more complex Neural Network.
{: .callout}
## Step 1. Formulate / Outline the problem: Penguin classification
In this episode we will be using the [penguin dataset](https://zenodo.org/record/3960218), this is a dataset that was published in 2020 by Allison Horst and contains data on three different species of the penguins.

The `palmerpenguins` data contains size measurements for three penguin species observed on three islands in the Palmer Archipelago, Antarctica.

> ## Goal
> The goal is to predict a penguins species using the attributes available in this dataset.
{: .objectives}

![Illustration of the three species of penguins found in the Palmer Archipelago, Antarctica: Chinstrap, Gentoo and Adele][palmer-penguins]
*Artwork by @allison_horst*

These data were collected from 2007 - 2009 by Dr. Kristen Gorman with the [Palmer Station Long Term Ecological Research Program](https://pal.lternet.edu/), part of the [US Long Term Ecological Research Network](https://lternet.edu/). The data were imported directly from the [Environmental Data Initiative](https://environmentaldatainitiative.org/) (EDI) Data Portal, and are available for use by CC0 license ("No Rights Reserved") in accordance with the [Palmer Station Data Policy](https://pal.lternet.edu/data/policies).


![Illustration of the beak dimensions called culmen length and culmen depth in the dataset][penguin-beaks]
*Artwork by @allison_horst*

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
{:.prereq}

### Create a notebook
Let us take a look at this dataset using a jupyter notebook.

Create a new notebook and call it assignment1.ipynb. We will start by importing the seaborn libraries that will help us get the dataset and visualize it.
Seaborn is a library for creating nice graphs and it includes the penguin dataset as a pandas dataframe to boot!
~~~
import seaborn as sns
~~~
{:.language-python}

### Loading the dataset
We can now load the penguin dataset using
~~~
penguins = sns.load_dataset('penguins')
~~~
{:.language-python}

This will give you a pandas dataframe which contains the penguin data

> ## penguin dataset
>
> Use seaborn to load the dataset and inspect the mentioned attributes.
> 1. What are the different features called?
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
> > It shows the target class is stored as a string.
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

The visualization library we will be using is called [Seaborn](https://seaborn.pydata.org/).
Seaborn is a powerful library with many visualizations. Keep in mind it requires the data to be in a
pandas dataframe, luckily our data is already in a dataframe
#### Pair Plot
One nice visualization for datasets with relatively few attributes is the Pair Plot.
This can be created using `sns.pairplot(...)`. It shows a scatterplot of each attribute plotted against each of the other attributes.
By using the `hue='class'` setting for the pairplot the graphs on the diagonal are layered kernel density estimate plots.

> ## Create the pairplot using Seaborn
> Use the seaborn pairplot function to create a pairplot
> * Is there any class that is easily distinguishable from the others?
> * Which combination of attributes shows the best separation?
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


[pairplot]: ../fig/pairplot.png "Pair Plot"
{: width="66%"}

[palmer-penguins]: ../fig/palmer_penguins.png "Palmer Penguins"
{: width="50%"}

[penguin-beaks]: ../fig/culmen_depth.png "Culmen Depth"
{: width="50%"}

## 2. Identify inputs and outputs

## Keras for Neural Networks
For this lesson we will be using [Keras](https://keras.io/) to define and train our neural network models.
Keras is a machine learning framework with ease of use as one of its main features.
It is part of the tensorflow python package and can be imported using `from tensorflow import keras`.





{% include links.md %}

