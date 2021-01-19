---
title: "Introduction"
teaching: 0
exercises: 0
questions:
- "What is deep learning?"
- "When is it used and not used?"
- "When is it successful?"
- "What is the workflow for deep learning?"
- "What are the tools involved?"
- "Why we chose Keras"

objectives:
- "State for which problems Deep learning is a useful tool"
- "List the available tools for Deep Learning"
- "List the steps of a Deep Learning workflow"
keypoints:
- "First key point. Brief Answer to questions. (FIXME)"
---

## What is Deep Learning?

*relation of DL to ML (infographics AI, ML, (NN), DL)

"deep stack of computations"

for complex problems

usually not possible to solve by human through set of rules

NN as one important model

* DL as one (!) tool of many

some nice starting info here: https://machinelearningmastery.com/what-is-deep-learning/

> ## History
>
> some benchmarks
> DL is not new as a concept but availability of comp resources is now at stage that it can be done efficiently
> some examples of interesting things that can be achieved using DL models
>
>
{: . callout}

> ## What is Deep Learning:
>
> Give multiple definitions of Deep Learning and discuss;
> OR 
> show both bubble representations of AI-ML-NN-DL and AI-ML-DL, ask which is right
>
{: .discussion}

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
1. formulate the problem

2. get data
where to get data from

3. prepare data
common steps for data preparation

4. choose model 
build from scratch vs reuse vs transfer learning

5. choose model parameters

6. train model

7. hyperparameter tuning

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

