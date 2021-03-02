---
title: "Introduction"
teaching: 0
exercises: 0
questions:
- "What is deep learning?"
- "When is it used and not used?"
- "When is it successful?"
- "What are the tools involved?"
- "What is the workflow for deep learning?"
objectives:
- "Understand the types of problems for which Deep Learning is useful to solve."
- "Recall some of the common tools available for Deep Learning."
- "Recall the steps of a Deep Learning workflow."
- "Understand why it is important to test the accuracy of deep learning system."
keypoints:
- "Keras is a deep learning framework that is easier to use than many of the alternatives."
---

# What is deep learning?

* Deep neural networks, analog of human brain
* Good at classification problems
* Needs lots of training data
* Outperforms many other machine learning techniques, especially on larger datasets

## What sort of problems can it solve?

* Pattern/object recognition
* Segmenting images (or any data)
* Translating between one set of data and another, for example natural language translation.
* Generating new data that looks similar to the training data, often used to create synthetic datasets, art or even "deepfake" videos.
  * This can also be used to give the illusion of enhancing data, for example making images look sharper, video look smoother or adding colour to black and white images. But beware of this, its not an accurate recreation of the original data, but a recreation based on something statistically similar, effectively a digital imagination of what that data could look like.

## What sort of problems can't it solve?

* Only small amounts of training data
* Tasks requiring an explanation of how the answer was arrived at
* Being asked to classify things which are nothing like their training data.


## What sort of problems can it solve, but shouldn't be used for?
Deep learning needs a lot of computational power, for this reason it often relies on specialist hardware like graphical processing units (GPUs). Many computational problems can be solved using less intensive techniques, but could still technically be solved with deep learning. 

The following could technically be achieved using deep learning, but it would be a wasteful way to do it:

* Logic operations, such as computing totals, averages, ranges etc. 
* Modelling well defined systems, where the equations governing them are known and understood.
* Basic computer vision tasks such as edge detection, decreasing colour depth or blurring an image.


> ## Deep Learning Problems Exercise
> Which of the following could you apply deep learning to?
> 1. Recognising whether or not a picture contains a bird.
> 2. Calculating the median and interquartile range of a dataset.
> 3. Identifying pictures of people when only one or two images of them are available.
> 4. Identifying people in pictures after being trained only on cats and dogs.
> 5. Translating English into French.
>> ## Solution
>> 1 and 5 are the sort of tasks often solved with deep learning. 2. technically possible but deep learning is massively overkill, you could do the same with much less computing power using traditional techniques. 3. will fail because there's not enough training data. 4. will fail because the deep learning system only knows what cats and dogs look like, it might accidentally classify the people as cats or dogs.
> {: .solution}
{: .challenge}

# Tools for Deep Learning

## TensorFlow

## PyTorch

## Caffe

## Keras

## Cloud based and abstracted tools

* Google Cloud Vision - https://cloud.google.com/vision
* Azure Compupter Vision - https://azure.microsoft.com/en-gb/services/cognitive-services/computer-vision/#features
* Amazon Rekognition - https://aws.amazon.com/rekognition/

# Deep Learning workflows

## Preparing Data

## Training

## Evaluating Training

## Classification

* Exercise on preparing data for deep learning, ask what changes might have to be made to some common data types.


{% include links.md %}

