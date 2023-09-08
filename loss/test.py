# Import the necessary libraries
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.tensorboard import SummaryWriter
import math
import matplotlib.pyplot as plt
from loss.partial_opt import MyAdam
from loss.partial_loss import MaskedCrossEntropyLoss
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 10)
        self.sf = nn.Softmax()

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.sf(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Loading the dataset
dataset = MNIST(root='data/', train=True, download=True, transform=ToTensor())
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
dataloader.dataset

sample_idx = torch.randint(len(dataloader), size=(1,)).item()
len(dataloader)

# Model
model = Net().to(device)
optimizer = MyAdam(model.parameters(), weight_decay=0.00001)
loss_fn = nn.CrossEntropyLoss()
# loss_fn = MaskedCrossEntropyLoss(alpha=0.05, beta=0.95)

num_epochs = 10
for i in range(num_epochs):
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step()

    plt.plot(i, loss.item(), 'ro-')
    print(i, '>> Loss :', loss.item())

plt.title('Losses over iterations')
plt.xlabel('iterations')
plt.ylabel('Losses')
plt.show()