---
title: "PyTorch basics"
teaching: 0
exercises: 0
questions:
- "Which framework choosing for deep learning?"
- "What is PyTorch?"
objectives:
- "Be aware of the various deep learning frameworks"
- "Get started with PyTorch"
keypoints:
- "PyTorch"
---

# Deep learning frameworks

It was worth taking some time to look inside and understand underlying concepts of neural 
networks but as we have experienced, writing neural networks with Python using 
[numpy](http://www.numpy.org/) is a bit cumbersome, especially when hidden layers are involved. 

For your research work, you will be mostly likely using high-level frameworks like Keras, TensorFlow 
or PyTorch. They will allow us to build very complex models quickly. 

## Most common high-level frameworks for deep learning

- [TensorFlow](https://www.tensorflow.org/) 
- [Theano](http://deeplearning.net/software/theano/)
- [Keras](https://keras.io/)
- [Torch](https://pytorch.org/)
- [Caffe](http://caffe.berkeleyvision.org/)


> ## Tips
> For more information see [presentation from Konstantin Shmelkov, Pauline Luc, Thomas Lucas, Vicky Kalogeiton, Stéphane Lathuilière on 
> Deep learning libraries](http://project.inria.fr/deeplearning/files/2016/05/DLFrameworks.pdf).
>
{: .callout}

### Which framework to choose when ..?

To choose, one need to look at:

- Code development (is it active?, what is the policy to release new versions, etc.)
- Performance (GPU/multi-GPU support, etc.)
- Installation/deployment (how difficult is it to install/test/deploy, etc.)
- **Community and documentation**

So why choosing [pyTorch](https://pytorch.org)?

![pytorch frontpage](../fig/pytorch_frontpage.png)

- Flexibility
	- Easy to extend - at any level, thanks to easy integration with C
		- Result :
			- whatever the problem, there is a package.
			- new generic bricks are often very rapidly implemented by the community and are easy to pull
	- Imperative (vs declarative)
	- Typical use case : write a new layer, with GPU implementation :
		a. Implement for CPU nn
		b. Implement for GPU cunn
		c. Test (jacobian and unit testing framework)
- Readability
- Modularity
- Speed
- Documentation is very good and the community is growing:
	- [Pytorch documentation](https://pytorch.org/docs/stable/index.html)
	- [Tutorials](https://pytorch.org/tutorials/)
	- [Examples](https://github.com/pytorch/examples)
	- [pytorch slack channel](https://pytorch.slack.com/) (send an email to slack@pytorch.org to get access)
	- [PyTorch discussions](https://discuss.pytorch.org/)
	- [Report bugs, request features, discuss issues and more](https://github.com/pytorch/pytorch)
	- [free online course with fast.ai](https://www.fast.ai/)

In short, **PyTorch is a very good framework for research**.


# Introduction to PyTorch


> ## Credits
> This lesson is taken from [Deep learning with PyTorch: a 60 minute blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html).
{: .callout}

## Tensors and variables

## Automatic differentiation for building and training neural networks (Autograd)

## Creating a simple Neural network with PyTorch

### The neural network class

### Training the network

### Testing the network

{% include links.md %}

