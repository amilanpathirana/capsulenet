import torchvision as tv
from torch.autograd import Variable
import torch

import utils
from capsnet import CapsNet

# Constants
BATCH_SIZE = 128
USE_GPU = True
N_CLASSES = 10
LOG_INTERVAL = 10
EPOCHS = 10


def train(epoch, model, dataloader, optim):
    model.train()

    for ix, (X, y) in enumerate(dataloader):
        target = utils.one_hot(y, model.routing_caps.n_unit)

        X, target = Variable(X), Variable(target)
        if USE_GPU:
            X, target = X.cuda(), target.cuda()

        y_hat = model(X)
        loss = model.loss(y_hat, target)
        loss.backward()
        optim.step()
        optim.zero_grad()

        if ix % LOG_INTERVAL == 0:
            # acc = utils.categorical_accuracy(y.float(), y_hat.cpu().data)
            print('[Epoch {}] ({}/{} {:.0f}%)\tLoss: {}'.format(
                epoch,
                ix * len(X),
                len(dataloader.dataset),
                100. * ix / len(dataloader),
                loss.data[0]
            ))

    return loss.data[0]


def test(epoch, model, dataloader):
    model.eval()
    loss_total = 0.
    for i, (X, y) in enumerate(dataloader):
        target = utils.one_hot(y, model.routing_caps.n_unit)

        X, target = Variable(X), Variable(target)
        if USE_GPU:
            X, target = X.cuda(), target.cuda()

        y_hat = model(X)
        loss = model.loss(y_hat, target)
        loss_total += loss.data[0]

    # acc = utils.categorical_accuracy(y.float(), y_hat.cpu().data)

    print('[EVAL Epoch {}] ({}/{} {:.0f}%) Loss: {} Accuracy: {}'.format(
        epoch,
        epoch * len(X),
        len(dataloader.dataset),
        100. * i / len(dataloader),
        loss_total / len(dataloader),
        '-'  # acc
    ))


trainloader, testloader = utils.mnist_dataloaders(
    '/home/erikreppel/data/mnist', BATCH_SIZE, USE_GPU)

model = CapsNet(n_conv_channel=256,
                n_primary_caps=8,
                primary_cap_size=1152,
                output_unit_size=16,
                n_routing_caps=3)

model = model.cuda() if USE_GPU else model
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print('Trainloader:', len(trainloader), len(trainloader.dataset))

for epoch in range(1, EPOCHS+1):
    train(epoch, model, trainloader, optimizer)
    test(epoch, model, testloader)
