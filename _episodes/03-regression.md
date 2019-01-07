---
title: "Regression with PyTorch"
teaching: 0
exercises: 0
questions:
objectives:
keypoints:
---

Let's start with a dummy dataset (radonmly generated data) and see how to build a neural network
with PyTorch.

# Input data

~~~
N = 64
D_in = 1000

# Create random Tensors to hold inputs and outputs
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)
~~~
{: .language-python}

We have 64 samples and each sample contains 1000 columns.

The goal is now to find a curve that describes best your dataset with neural network.

# Neural network class definition

Our multi-layered neural network will have one hidden layer (it is still not deep learning!) so in total 3 layers:
- one input layer
- one hidden layer
- one output layer

We still use Linear model for each layer and we take a ReLU (Rectified Linear Unit) Activation Function
to be applied one the hidden layer:

~~~
import torch
class Model(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super().__init__()
        self.l1 = torch.nn.Linear(D_in, H)
        self.relu = torch.nn.ReLU()
        self.l2=torch.nn.Linear(H, D_out)

    def forward(self, X):
        return self.l2(self.relu(self.l1(X)))
~~~
{: .language-python}


# Training

~~~
# Define 100 neurons in the hidden layer
H = 100
# and 10 neurons in the output layer
D_out = 10
model = Model(D_in, H, D_out)

loss_fn = torch.nn.MSELoss(reduction='sum')

# Use the optim package to define an Optimizer that will update the weights of
# the model for us. Here we will use Adam; the optim package contains many other
# optimization algoriths. The first argument to the Adam constructor tells the
# optimizer which Tensors it should update.
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for t in range(500):
    # Forward pass: compute predicted y by passing x to the model.
    y_pred = model(x)

    # Compute and print loss.
    loss = loss_fn(y_pred, y)
    print(t, loss.item())

    # Before the backward pass, use the optimizer object to zero all of the
    # gradients for the variables it will update (which are the learnable
    # weights of the model). This is because by default, gradients are
    # accumulated in buffers( i.e, not overwritten) whenever .backward()
    # is called. Checkout docs of torch.autograd.backward for more details.
    optimizer.zero_grad()

    # Backward pass: compute gradient of the loss with respect to model
    # parameters
    loss.backward()

    # Calling the step function on an Optimizer makes an update to its
    # parameters
    optimizer.step()
~~~
{: .language-python}

# Test/validation
~~~
y_last = model(x)
print(y_last - y)

~~~
{: .language-python}

> ## Challenge
>
> Change your neural network and your dataset:
> - try different values for the learning rate
> - add more samples and generate a training and testing dataset
> - Add a new hidden layer
>
{: .challenge}

{% include links.md %}
