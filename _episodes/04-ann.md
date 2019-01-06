---
title: "Artificial Neural Network (ANN)"
teaching: 0
exercises: 0
questions:
- "How to prepare data?"
- "How to load data?"
- "How to run your ANN with your data?"
objectives:
- "Learn how to prepare your data for PyTorch"
- "Define your neural network and train it with your data"
- "Test your neural network with your test data"
keypoints:
- "ANN"
---

# Prepare your Data

## Training data versus Test data

## Normalization

# Load your data
Let's use the MNIST dataset and download it with PyTorch. We define sizes of the training and test batches. 
~~~
import torch
import torchvision
torch.random.manual_seed(1)

batch_size_train = 64
batch_size_test = 1000

train_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST(root='./data/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./data/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_test, shuffle=True)
  
~~~
{: .language-python}
# Define your network
~~~
class ANN(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(28*28, 300)   #input size
        self.fc2 = torch.nn.Linear(300,300)
        self.fc3 = torch.nn.Linear(300,300)
        self.fc4 = torch.nn.Linear(300, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)   # the size -1 is inferred from other dimensions (28*28)
        x = torch.sigmoid(self.fc1(x))
        x = torch.nn.functional.dropout(x, training=self.training)
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        x = self.fc4(x)
        return x          
~~~
{: .language-python}

~~~
network = ANN()
optimiser = torch.optim.Adam(network.parameters(), lr=learning_rate,
                      weight_decay=0.005) 
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser,mode='min')
criterion = torch.nn.CrossEntropyLoss()         

~~~
{: .language-python}

# Train your neural network
~~~
def train(epoch):
    network.train()
    training_loss = 0
    train_corrects = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        optimiser.zero_grad()
        output = network(data)
        _, preds = torch.max(output.data, 1)
        loss = criterion(output, target)
        loss.backward()
        optimiser.step()
        training_loss += loss.item()
        train_corrects += torch.sum(preds == target.data) 

        train_losses.append(loss.item())
        train_counter.append((batch_idx*batch_size_train) + ((epoch-1)*len(train_loader.dataset)))
        
    torch.save(network.state_dict(), 'model.pth')
    torch.save(optimiser.state_dict(), 'optimiser.pth')
  
    epoch_acc= train_corrects.double() / len(train_loader.dataset)
    training_losses.append(training_loss/len(train_loader.dataset))
    print('Epoch {} , Average training loss is {:.6f} and accuracy is {}/{} {:.0f}%'.format((epoch+1),
                        training_loss/len(train_loader),train_corrects.double(),
                                        len(train_loader.dataset),epoch_acc*100.))
~~~
{: .language-python}

# Test your neural network
~~~
def test():
    network.eval()
    test_loss = 0
    test_corrects = 0
    total= 0
    with torch.no_grad():
        for batch_idx,(data, target) in enumerate(test_loader):
            output = network(data)
            _, preds = torch.max(output.data, 1)
            test_loss += criterion(output, target).item()
            total += target.size(0)
            test_corrects += torch.sum(preds == target.data) 
        
        epoch_acc= test_corrects.double() / len(test_loader.dataset)
        test_loss /= len(test_loader)
        test_losses.append(test_loss)
       
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, test_corrects, len(test_loader.dataset),
            100. * epoch_acc))
        scheduler.step(test_loss)
~~~
{: .language-python}
# Run train and test for network
~~~
train_losses = []
train_counter = []
test_losses = []
training_losses=[]
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]
n_epochs = 10
test()
for epoch in range(n_epochs):
    train(epoch)
    test()
~~~
{: .language-python}

{% include links.md %}

