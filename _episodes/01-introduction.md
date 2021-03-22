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
- "Why we chose Keras"
objectives:
- "State for which problems Deep learning is a useful tool"
- "List the available tools for Deep Learning"
- "List the steps of a Deep Learning workflow"
- "Understand why it is important to test the accuracy of deep learning system."
keypoints:
- "Keras is a deep learning framework that is easier to use than many of the alternatives."
---

## What is Deep Learning?


* Deep neural networks, analog of human brain
* Good at classification problems
* Needs lots of training data
* Outperforms many other machine learning techniques, especially on larger datasets


* relation of DL to ML (infographics AI, ML, (NN), DL)
![AI-ML-DL](../fig/AI_ML_DL_bubble_square_draft.png)

* "deep stack of computations"

* for complex problems, usually not possible to solve by human through set of rules

* DL as one (!) tool of many

* Prerequisite: lots of Data

![ML_DL](../fig/ML_DL_draft.png)


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


> ## Application areas/History
>
> some benchmarks
> DL is not new as a concept but availability of comp resources is now at stage that it can be done efficiently
> some examples of interesting things that can be achieved using DL models:
> image classification, text generation, language translation, GAN, Gameplay (Dota/Go), voice recognition,
> self-driving cars, natural language and image processing, predictive forecasting, fraud detection in financial applications,
> financial time-series forecasting, predictive and prescriptive analytics, medical image processing, power systems research,
> recommendation systems
> we are still far from computers understanding us
>
{: . callout}

> ## What is Deep Learning:
>
> Give multiple definitions of Deep Learning and discuss;
> OR
> show both bubble representations of AI-ML-NN-DL and AI-ML-DL, ask which is right
>
{: .discussion}


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

> ## What is DL - solution
>
> learning outcome -> it is vague, find your own
> OR
> hah! both are right in some ways, explain why NN has its place there as well
>
{: .solution}

*the neuron/perceptron/unit/

* inputs, weight, sum, non-linearity, output
* -> DL: stacked perceptrons

* training challenges
  * gradient
  * overfitting

## Building blocks of NN

*building blocks of a DL model:
    *example case CNN
        *show how each layer may look
        *eg based on face features
        *we learn from many examples what and eye looks like and where it is located
*-> needs high amount of data

## Deep Learning workflow

(one way, open for discussion and to be filled with information)
1. Formulate/ Outline the problem

2. Identify inputs and outputs

3. Prepare data
common steps for data preparation

4. Choose a cost function and metrics

5. Choose a pretrained model or start building architecture from scratch
build from scratch vs reuse vs transfer learning

6. Train model

7. Tune hyperparameters

8. 'predict'

(1 and 2 sometimes mixed)

> ## DL workflow
>
> Give an example problem: create DL workflow
>
{: .challenge}

> ## DL workflow - solution
>
>
{: .solution}

## Deep Learning in Python

* many different options
    * name them (and known limitations)

* why keras (and why 'with' tensor-flow)
    * easy start
    * huge community
        * help and tutorials online
    *

> ## Keras
>
> load keras and do something simple with it, get familiar with keras docs
>
{: .challenge}


> ## package and enviroment management
>
> some notes on importance of package and environment management for DL
> especially when trying out different setups
>
{: . callout}

> ## Computational resources
>
> some notes on computational resources needed for DL
> we try to have everyhting in the course run on a laptop with minimum requirements xxx
> some links where to get more resources to work with, eg HPC3Europe,AWS?,..?
>
{: . callout}

> ## Importance of data mining
>
> some notes on data mining
> why important to know your data for DL
> also mention that some models have prerequisites of data distribution etc
>
>
{: . callout}


FIXME

{% include links.md %}

