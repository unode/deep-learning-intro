---
title: "Introduction"
teaching: 0
exercises: 0
questions:
- "What is deep learning?"
- "When does it make sense to use and not use deep learning?"
- "When is it successful?"
- "What are the tools involved?"
- "What is the workflow for deep learning?"
- "Why we chose Keras"
objectives:
- "Recall the sort of problems for which Deep learning is a useful tool"
- "List some of the available tools for Deep Learning"
- "Recall the steps of a Deep Learning workflow"
- "Understand why it is important to test the accuracy of deep learning system."
- "Identify the inputs and outputs of a deep learning system."
- "Test that we've correctly installed the Keras, Seaborn and Sklearn libraries"
keypoints:
- "Machine learning is the process where computers learn to recognise patterns of data."
- "Artificial neural networks are a machine learning technique based on a model inspired by groups of neurons in the brain."
- "Artificial neural networks can be trained on example data."
- "Deep learning is a machine learning technique based on using many artificial neurons arranged in layers."
- "Deep learning is well suited to classification and prediction problems such as image recognition."
- "To use deep learning effectively we need to go through a workflow of: defining the problem, identifying inputs and outputs, preparing data, choosing the type of network, choosing a loss function, training the model, tuning Hyperparameters, measuring performance before we can classify data."
- "Keras is a deep learning library that is easier to use than many of the alternatives such as TensorFlow and PyTorch."
---

## What is Deep Learning?


### Deep Learning, Machine Learning and Artificial Intelligence

Deep learning (DL) is just one of many techniques collectively known as machine learning. Machine learning (ML) refers to techniques where a computer can "learn" patterns in data, usually by being shown numerous examples to train it. People often talk about machine learning being a form of artificial intelligence (AI). Definitions of artificial intelligence vary, but usually involve having computers mimic the behaviour of intelligent biological systems. Since the 1950s many works of science fiction have dealt with the idea of an artificial intelligence which matches (or exceeds) human intelligence in all areas. Although there have been great advances in AI and ML research recently we can only come close to human like intelligence in a few specialist areas and are still a long way from a general purpose AI.


#### Neural Networks

A neural network is an artificial intelligence technique loosely based on the way neurons in the brain work. A neural network consists of connected computational units called neurons. Each neuron takes the sum of all its inputs, performs some, typically non-linear, calculation on them and produces one output. This calculation is called the activation function. The connections between neurons are called edges, these edges typically have a weight associated with them. This weight determines the 'strength' of the connection, these weights are adjusted during training. In this way, the combination of neurons and edges describe a computational graph, an example can be seen in the image below. In most neural networks neurons are aggregated into layers. Signals travel from the input layer to the output layer, possibly through one or more intermediate layers called hidden layers.

Neural networks aren't a new technique, they have been around since the late 1940s. But until around 2010 neural networks tended to be quite small, consisting of only 10s or perhaps 100s of neurons. This limited them to only solving quite basic problems. Around 2010 improvements in computing power and the algorithms for training the networks made much larger and more powerful networks practical. These are known as deep neural networks or deep learning.

![An infographics showing the relation of AI, ML, NN and DL](../fig/AI_ML_DL_bubble_square_draft.png)

Deep learning requires extensive training using example data which shows the network what output it should produce for a given input. One common application of deep learning is classifying images. Here the network will be trained by being "shown" a series of images and told what they contain. Once the network is trained it should be able to take another image and correctly classify its contents. But we are not restricted to just using images, any kind of data can be learned by a deep learning neural network. This makes them able to appear to learn a set of complex rules only by being shown what the inputs and outputs of those rules are instead of being taught the actual rules. Using these approaches deep learning networks have been taught to play video games and even drive cars. The data on which networks are trained usually has to be quite extensive, typically including thousands of examples. For this reason they are not suited to all applications and should be considered just one of many machine learning techniques which are available.

The image below shows the architecture of a traditional "shallow" network (top) and a deep network (bottom). In the shallow network we have to do some extra pre-processing of the data to make it suitable to for the network to understand it. Each circle represents one neuron in the network and the lines the edges connecting them. In both cases the final (right most) layer of the network outputs a zero or one to determine if the input data belongs to the class of data we're interested in.

[//]: # "![An example neural network with ][neural-network][*Glosser.ca, CC BY-SA 3.0 <https://creativecommons.org/licenses/by-sa/3.0>, via Wikimedia Commons*](https://commons.wikimedia.org/wiki/File:Colored_neural_network.svg)"

![An example of a neural network](../fig/ML_DL_draft.png)


### What sort of problems can deep learning solve?

* Pattern/object recognition
* Segmenting images (or any data)
* Translating between one set of data and another, for example natural language translation.
* Generating new data that looks similar to the training data, often used to create synthetic datasets, art or even "deepfake" videos.
  * This can also be used to give the illusion of enhancing data, for example making images look sharper, video look smoother or adding colour to black and white images. But beware of this, its not an accurate recreation of the original data, but a recreation based on something statistically similar, effectively a digital imagination of what that data could look like.

### What sort of problems can't deep learning solve?

* Where only small amounts of training data is available.
* Tasks requiring an explanation of how the answer was arrived at.
* Being asked to classify things which are nothing like their training data.

### What sort of problems can deep learning solve, but shouldn't be used for?

Deep learning needs a lot of computational power, for this reason it often relies on specialist hardware like graphical processing units (GPUs). Many computational problems can be solved using less intensive techniques, but could still technically be solved with deep learning.

The following could technically be achieved using deep learning, but it would probably be a very wasteful way to do it:

* Logic operations, such as computing totals, averages, ranges etc. (see [this example](https://joelgrus.com/2016/05/23/fizz-buzz-in-tensorflow) applying deep learning to solve the "FizzBuzz" problem often used for programming interviews)
* Modelling well defined systems, where the equations governing them are known and understood.
* Basic computer vision tasks such as edge detection, decreasing colour depth or blurring an image.

> ## Deep Learning Problems Exercise
>
> Which of the following could you apply deep learning to?
> 1. Recognising whether or not a picture contains a bird.
> 2. Calculating the median and interquartile range of a dataset.
> 3. Identifying MRI images of a rare disease when only one or two example images available for training.
> 4. Identifying people in pictures after being trained only on cats and dogs.
> 5. Translating English into French.
>
> > ## Solution
> >
> > 1.  and 5 are the sort of tasks often solved with deep learning.
> > 2. is technically possible but deep learning is massively overkill, you could do the same with much less computing power using traditional techniques.
> > 3. will probably fail because there's not enough training data.
> > 4. will fail because the deep learning system only knows what cats and dogs look like, it might accidentally classify the people as cats or dogs.
> {: .solution}
{: .challenge}



## Deep Learning workflow

To apply deep learning to a problem there are several steps we need to go through:

### 1. Formulate/ Outline the problem

Firstly we must decide what it is we want the our deep learning system to do. Is it going to classify some data into one of a few categories? For example if we have an image of some hand written characters, the neural network could classify which character it is being shown. Or is it going to perform a prediction? For example trying to predict what the price of something will be tomorrow given some historical data on pricing and current trends.

[//]: # "What about pattern association tasks like language translation?"

### 2. Identify inputs and outputs

Next we need to identify what the inputs and outputs of the neural network will be. This might require looking at our data and deciding what features of the data we can use as inputs. If the data is images then the inputs could be the individual pixels of the images. Depending upon the data we're dealing with it might help to visualize the data or perform some kind of exploration to discover what features we have available.

For the outputs we'll need to look at what we want to identify from the data. If we are performing a classification problem then typically we will have one output for each potential class. The classification that the network identifies will have its output set to one and the other classes will output zero. Some networks might make this a probability and use numbers between zero and one to indicate the probability of each class.


### 3. Prepare data

Many datasets aren't ready for immediate use in a neural network and will require some preparation. Neural networks can only really deal with numerical data, so any non-numerical data (for example words) will have to be somehow converted to numerical data. Many datasets will also have missing data and we will need to clean up our data and remove items with missing data or find a way to convert them to sensible values.

Next we'll need to divide the data into multiple sets. One of these will be used by the training process and we'll call it the training set. Another will be used to evaluate the accuracy of the training and we'll call that one the test set. Sometimes we'll also use a 3rd set known as a validation set to check our results after training is complete. Typically 80-90% of the data will be used as the training set and 10-20% for the test set and another 10% for the validation set (if we have one). We should split the data in such a way that the training and test set are similar, for example by randomising which items go in which set instead of just taking the first 90% as the training set and the remaining 10% as the test set, especially if the last 10% of the data is very different to the previous 90%.

### 4. Choose a pre-trained model or build a new architecture from scratch

Often we can use an existing neural network instead of designing one from scratch. There are a number of well publicised networks which have been shown to perform well at certain tasks, if you know of one which already does a similar task well then it makes sense to use it. This concept can be taken one step further, somebody else might have released a pre-trained network which is already trained on similar data and that can work for our purposes. If there's a network which was trained on similar data but might need a little bit more training on our specific data then this is possible using a technique called transfer learning which we'll talk about in more detail later.

If instead we decide we do want to design our own network then we need to think about how many input neurons it will have, how many hidden layers and how many outputs, what types of layers we use (we'll explore the different types later on). This will probably need some experimentation and we might have to try tweaking the network design a few times before we see acceptable results.


### 5. Choose a loss function and metrics

The loss function tells the training algorithm how far away the predicted value was from the true value. There are a number of different types of function we can use here that might be suited to different types of data and network architectures. We'll look at choosing a loss function in more detail later on.


### 6. Train the model

Before we can train the model we have one final component to decide upon, the optimizer. The optimizer is responsible for taking the output of the loss function and then applying some changes to weights within the network. We can think of the training process as like trying to gradually descend a (very jagged looking) curve on a graph (that looks more like a rocky mountain side than a smooth curve) and find the bottom of it, without knowing in advance where that bottom is. Along the way we might hit some flat or even rising parts of the curve that make it look like we've reached the bottom when we haven't, we call these local minima. Different optimizers and optimizer parameters can help prevent us getting stuck in local minima. As with the loss function the best choice may depend on the kind of data and network architecture that we're using.

Now that we've set the optimizer function we can go ahead and start training our network. We'll probably keep doing this for a given number of iterations or epochs or until the loss function gives a value under a certain threshold.

[//]: # "is a graph of a reduction in loss sensible here?"

### 7. Tune Hyperparameters

Hyperparameters are all the parameters set by the person configuring the machine learning instead of those learned by the algorithm itself. The Hyperparameters include the number of epochs or the parameters for the optimizer. If we are really trying to optimize things then we might even do a sweep through a whole range of Hyperparameters and use the loss function to decide which Hyperparameters performed best.

### 8. Measure Performance

Once we think the network is performing well we want to measure its performance. To do this we might use the validation set of data that we put aside earlier and didn't use as part of the training process. For a classification task we can often evaluate four different measures for each possible class in the data:

1. True Positives: Where we correctly classified something as belonging to this class.
2. True Negatives: Where we correctly classified that something wasn't a member of this class.
3. False Positives: Where we incorrectly classified something as belonging to this class when it doesn't.
4. False Negatives: Where we incorrectly classified something as not belonging to this class when it does.

There are a number of metrics which combine some of these four measures. For example accuracy is defined as the number of true positives plus the number of true Negatives divided by the total number of samples (TP+TN/n). Exactly which of these we want to optimize may depend upon the task we are trying to achieve and how acceptable miss-classification is.

### 9. Perform a Prediction/Classification

Now that we have a trained network that performs at a level we are happy with we can go and use it on real data to perform a prediction. At this point we might want to consider publishing a file with both the architecture of our network and the weights which it has learned (assuming we didn't use a pre-trained network). This will allow others to use it as as pre-trained network for their own purposes.


> ## Deep Learning workflow exercise
>
> Think about a problem you'd like to use deep learning to solve.
> 1. What do you want a deep learning system to be able to tell you?
> 2. What data inputs and outputs will you have?
> 3. Do you think you'll need to train the network or will a pre-trained network be suitable?
> 4. What data do you have to train with? What preparation will your data need? Consider both the data you are going to predict/classify from and the data you'll use to train the network.
>
> Discuss your answers with the group or the person next to you.
{: .challenge}

## Deep Learning Libraries

There are many software libraries available for deep learning including:

### TensorFlow

[TensorFlow](https://www.tensorflow.org/) was developed by Google and is one of the older deep learning libraries, first released to the public in 2015. It was originally developed for the C++ programming language and doesn't always follow a "pythonic" style that might make it seem a bit different to other libraries in Python. Its been ported to many other languages including Java, JavaScript, Ruby and Swift. It works on a very low level of individual "tensors" or arrays. Its capable of much more than deep learning and offers a very versatile toolkit for tensor operations. As a result it often takes a lot more lines of code to write deep learning operations in TensorFlow than other libraries. It offers (almost) seamless integration with GPU accelerators and Google's own TPU (Tensor Processing Unit) chips that are built specially for machine learning.

### PyTorch

[PyTorch](https://pytorch.org/) was developed by Facebook in 2016 and is a popular choice for deep learning applications. It was developed for Python from the start and feels a lot more "pythonic" than TensorFlow. Like TensorFlow it was designed to do more than just deep learning and offers some very low level interfaces. Like TensorFlow it's also very easy to integrate PyTorch with a GPU. In many benchmarks it out performs the other libraries.

### Keras

[Keras](https://keras.io/) is designed to be easy to use and usually requires fewer lines of code than other libraries. We have chosen it for this workshop for that reason. Keras can actually work on top of TensorFlow (and several other libraries), hiding away the complexities of TensorFlow while still allowing you to make use of their features.

The performance of Keras is sometimes not as good as other libraries and if you are going to move on to create very large networks using very large datasets then you might want to consider one of the other libraries. But for many applications the performance difference will not be enough to worry about and the time you'll save with simpler code will exceed what you'll save by having the code run a little faster.

Keras also benefits from a very good set of [online documentation](https://keras.io/guides/) and a large user community. You will find that most of the concepts from Keras translate very well across to the other libraries if you wish to learn them at a later date.

### Installing Keras and other dependencies

Follow the instructions in the [setup]({{ page.root }}//setup) document to install Keras, Seaborn and Sklearn.

> ## Testing Keras Installation
> Lets check you have a suitable version of Keras installed.
> Open up a new Jupyter notebook or interactive python console and run the following commands:
> ~~~
> from tensorflow import keras
> print(keras.__version__)
> ~~~
> {:.language-python}
> > ## Solution
> > You should get a version number reported. At the time of writing 2.4.0 is the latest version.
> > ~~~
> > 2.4.0
> > ~~~
> > {:.output}
> {:.solution}
{:.challenge}

> ## Testing Seaborn Installation
> Lets check you have a suitable version of seaborn installed.
> In your Jupyter notebook or interactive python console run the following commands:
> ~~~
> seaborn
> print(seaborn.__version__)
> ~~~
> {:.language-python}
> > ## Solution
> > You should get a version number reported. At the time of writing 0.11.1 is the latest version.
> > ~~~
> > 0.11.1
> > ~~~
> > {:.output}
> {:.solution}
{:.challenge}


> ## Testing Sklearn Installation
> Lets check you have a suitable version of sklearn installed.
> In your Jupyter notebook or interactive python console run the following commands:
> ~~~
> import sklearn
> print(sklearn.__version__)
> ~~~
> {:.language-python}
> > ## Solution
> > You should get a version number reported. At the time of writing 0.24.1 is the latest version.
> > ~~~
> > 0.24.1
> > ~~~
> > {:.output}
> {:.solution}
{:.challenge}


{% include links.md %}
