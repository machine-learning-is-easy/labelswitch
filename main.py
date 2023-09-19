# -*- coding: utf-8 -*-
import torch
import torchvision
import torchvision.transforms as transforms
# import matplotlib.pyplot as plt
from torch.utils.data import random_split
import torch.optim as optim
import torch.nn as nn
from loss.resonance import Resonance
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.models as models
import os
import pandas as pd
from model_define import *
from model_define.googlenet import googlenet, GoogleNet
from model_define.resnet import ResNet, resnet18
torch.manual_seed(100)
torch.cuda.manual_seed(100)
import random
random.seed(100)
import argparse

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))
parser = argparse.ArgumentParser(description='Train Model')
parser.add_argument('--epoch', '-e', dest='epoch', default=50, help='epoch')
parser.add_argument('--dataset', '-d', dest='dataset', default="CIFAR10", help='dataset', required=False)
parser.add_argument('--model', '-m', dest='model', default="CIFARNet", help='model', required=False)
parser.add_argument('--opt_alg', '-a', dest='opt_alg', default="ADAM", help='opt_alg', required=False)
parser.add_argument('--lr', '-lr', dest='lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--resonance', '-pt', dest='resonance', default="no_res", help='no_res|init_res|full_res')


args = parser.parse_args()
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

########################################################################
# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1].

########################################################################
# .. note::
#     If running on Windows and you get a BrokenPipeError, try setting
#     the num_worker of torch.utils.data.DataLoader() to 0.


batch_size = 128

current_folder = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_folder, 'data')

def define_model(model_type, num_class):
    if model_type.lower() == "googlenet":
        net = googlenet(num_class)
    elif model_type.lower() == 'resnet':
        net = ResNet(num_class)
    else:
        raise Exception("Unable to support model type of {}".args.model)

    return net

def reinitialization_model(model):
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

if args.dataset == "CIFAR10":

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root=data_path, train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=0)

    testset = torchvision.datasets.CIFAR10(root=data_path, train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=0)
    num_channel = 3
    image_size = trainset.data.shape[1]
    dataclasses_num = len(trainset.classes)

elif args.dataset == "CIFAR100":
    transform = transforms.Compose([transforms.ToTensor()])

    trainset = torchvision.datasets.CIFAR100(root=data_path, train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=0)

    testset = torchvision.datasets.CIFAR100(root=data_path, train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=0)
    num_channel = 3
    image_size = trainset.data.shape[1]
    dataclasses_num = len(trainset.classes)

net = define_model(args.model, dataclasses_num)
net = net.to(device)
criterion = nn.CrossEntropyLoss()


########################################################################
# 2. Define a Convolutional Neural Network
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Copy the neural network from the Neural Networks section before and modify it to
# take 3-channel images (instead of 1-channel images as it was defined).


def defineopt(model):
    if args.opt_alg == "ADAM":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    elif args.opt_alg == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
    elif args.opt_alg == "RADAM":
        optimizer = optim.RAdam(model.parameters(), lr=args.lr)
    elif args.opt_alg == "ADADELTA":
        optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    elif args.opt_alg == "LWADAM":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    else:
        raise Exception("Not accept optimizer of {}".args.opt_alg)
    return optimizer


optimizer = defineopt(net)
def define_scheduler(optimizer):
    return ReduceLROnPlateau(optimizer, patience=10)
scheduler = define_scheduler(optimizer)
########################################################################
def run_test(net):
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            # calculate outputs by running images through the network
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            outputs = net(images)
            if resonance:
                outputs = resonance(outputs)
            # the class with the highest energy is what we choose as prediction
            predicted = torch.argmax(outputs.data, 1)
            total += labels.size(0)
            labels = labels.to(device)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total


# 4. Train the network
for t in range(10):  # train model 10 times
    if args.resonance and args.resonance == "init_res":
        resonance = Resonance(dataclasses_num, device=device)
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            try:
                outputs = net(inputs)
            except Exception as ex:
                raise Exception("Inference model encounter Exceptions")
            resonance.buffer(outputs, labels)
        resonance.create_map()
    else:
        resonance = None

    acc = []
    for epoch in range(0, int(args.epoch)):  # loop over the dataset multiple times
        running_loss = 0.0
        if args.resonance == "full_res":
            if epoch % 2 == 0:
                resonance = Resonance(dataclasses_num, device=device)
                for i, data in enumerate(trainloader, 0):
                    inputs, labels = data
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    try:
                        outputs = net(inputs)
                    except Exception as ex:
                        raise Exception("Inference model encounter Exceptions")
                    resonance.buffer(outputs, labels)
                resonance.create_map()

        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]

            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            try:
                outputs = net(inputs)
            except Exception as ex:
                raise Exception("Inference model encounter Exceptions")

            if resonance:
                outputs = resonance(outputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # zero the parameter gradients
            optimizer.zero_grad()
            running_loss += loss.item()
        model_path = os.path.join(current_folder, 'model', '{}_{}_{}_net.pth'.format(args.dataset, args.model, args.resonance))
        acc_epoch = run_test(net)
        scheduler.step(metrics=acc_epoch)
        acc_epoch = round(acc_epoch, 2)
        acc.append([epoch, acc_epoch, round(running_loss, 2)])
        print("{} time {} epoch acc is {}".format(t, epoch, acc_epoch))
    result_file = os.path.join(os.path.join(current_folder, 'result', '{}_{}_{}_result'.format(args.dataset, args.model, args.resonance), "{}.csv".format(str(t))))
    if not os.path.exists(os.path.dirname(result_file)):
        os.makedirs(os.path.dirname(result_file))
    pd.DataFrame(acc).to_csv(result_file, header=["epoch", "training_acc", "training_loss"], index=False)
    if isinstance(net, ResNet):
        del net
        del optimizer
        net = resnet18(dataclasses_num)
    elif isinstance(net, GoogleNet):
        del net
        del optimizer
        net = define_model(args.model, dataclasses_num)
    else:
        raise Exception("Unable to accept the net type")

    net = net.to(device)
    optimizer = defineopt(net)
    scheduler = define_scheduler(optimizer)
