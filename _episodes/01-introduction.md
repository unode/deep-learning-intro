---
title: "Introduction"
teaching: 0
exercises: 0
questions:
- "What is deep learning?"
- "What is a computational graph?"
- "What is a neural network?"
- "How to start deep learning with Python?"
objectives:
- "Understand what is deep learning"
- "Learn about simple neural network"
- "Create a simple neural network in Python"
keypoints:
- "deep learning"
- "computational graph"
- "neural network"
---

# What, Why and How?

A short introduction on [deep learning](https://cicero.xyz/v3/remark/0.14.0/github.com/UiOHive/deep-learning_intro/gh-pages/intro.md/#1).

> ## Keep it in mind
> 
> <img src="../fig/Why-Deep-Learning.png" style="width: 550px;"/>
>
> <p style="font-size:12px"><i>source</i>: <a href="https://www.slideshare.net/ExtractConf">Andrew Ng</a>, all rights reserved.</p>
>
{: .callout}


# From a neuron to neural networks

# Computational graph

# Create your first neural network in Python
~~~
## Neural network with numpy for regression
import numpy as np
#Input array
X=np.array([[1,0,1,0],[1,0,1,1],[0,1,0,1]])
#Output
y=np.array([[1],[1],[0]])

#Sigmoid Function
def sigmoid (x):
     return 1/(1 + np.exp(-x))

#Derivative of Sigmoid Function
def sigmoid_derivative(x):
     return x * (1 - x)

# initialization of parameters
epoch=5000 #amount of training iterations
lr=0.1 #Setting learning rate
inputlayer_neurons = X.shape[1] #number of features in data set
hiddenlayer_neurons = 3 #number of neurons in the hidden layer
output_neurons = 1 #number of neurons at output layer

#weight and bias initialization
wh=np.random.uniform(size=(inputlayer_neurons,hiddenlayer_neurons))
bh=np.random.uniform(size=(1,hiddenlayer_neurons))
wout=np.random.uniform(size=(hiddenlayer_neurons,output_neurons))
bout=np.random.uniform(size=(1,output_neurons))

for i in range(epoch):
#Forward Propagation
    hidden_layer_input1=np.dot(X,wh)
    hidden_layer_input=hidden_layer_input1 + bh
    hiddenlayer_activations = sigmoid(hidden_layer_input)
    output_layer_input1=np.dot(hiddenlayer_activations,wout)
    output_layer_input= output_layer_input1+ bout
    output = sigmoid(output_layer_input)

  #Backpropagation
    E = y-output
    slope_output_layer = sigmoid_derivative(output)
    slope_hidden_layer = sigmoid_derivative(hiddenlayer_activations)
    d_output = E * slope_output_layer
    Error_at_hidden_layer = d_output.dot(wout.T)
    d_hiddenlayer = Error_at_hidden_layer * slope_hidden_layer
    wout += hiddenlayer_activations.T.dot(d_output) *lr
    bout += np.sum(d_output, axis=0,keepdims=True) *lr
    wh += X.T.dot(d_hiddenlayer) *lr
    bh += np.sum(d_hiddenlayer, axis=0,keepdims=True) *lr

print('actual :\n', y, '\n')
print('predicted output :\n', output)
~~~

# Overall code layout


{% include links.md %}

