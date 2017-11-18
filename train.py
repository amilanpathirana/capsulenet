from torch.autograd import Variable
import torch
import time
from os import path

from args import args
import utils
from log import Logger
from capsnet import CapsNet

# Constants
logger = Logger(args.batch_size, args.visdom)
logger.plain('Training CapsNet with the following settings:\n{}'.format(args))

if args.dataset == 'MNIST':
    N_CLASSES = 10
else:
    raise Exception('Invalid dataset')

start = time.strftime("%Y-%m-%d-%H:%M")


def train(epoch, model, dataloader, optim):
    model.train()

    for ix, (X, y) in enumerate(dataloader):
        target = utils.one_hot(y, model.final_caps.n_unit)

        X, target = Variable(X), Variable(target)
        if args.use_gpu:
            X, target = X.cuda(), target.cuda()

        y_hat = model(X)
        loss = model.loss(y_hat, target)
        loss.backward()
        optim.step()
        optim.zero_grad()

        if ix % args.log_interval == 0:
            preds = model.capsule_prediction(y_hat)
            acc = utils.categorical_accuracy(y.float(), preds.cpu().data)
            # acc = utils.categorical_accuracy(y.float(), y_hat.cpu().data)
            logger.log(epoch, ix, len(dataloader.dataset), start+'_TRAIN',
                       loss=loss.data[0], acc=acc)

    return loss.data[0]


def test(epoch, model, dataloader):
    model.eval()

    for i, (X, y) in enumerate(dataloader):
        target = utils.one_hot(y, model.final_caps.n_unit)

        X, target = Variable(X), Variable(target)
        if args.use_gpu:
            X, target = X.cuda(), target.cuda()

        y_hat = model(X)
        loss = model.loss(y_hat, target)

        preds = model.capsule_prediction(y_hat)
        acc = utils.categorical_accuracy(y.float(), preds.cpu().data)
        logger.log(epoch, i, len(dataloader.dataset), start+'_TEST',
                   loss=loss.data[0], acc=acc)


trainloader, testloader = utils.mnist_dataloaders(args.data_path,
                                                  args.batch_size,
                                                  args.use_gpu)

model = CapsNet(n_conv_channel=256,
                n_primary_caps=8,
                primary_cap_size=1152,
                output_unit_size=16,
                n_routing_iter=3)

if args.load_checkpoint != '':
    model.load_state_dict(torch.load(args.load_checkpoint))

model = model.cuda() if args.use_gpu else model
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

for epoch in range(1, args.epochs+1):
    train(epoch, model, trainloader, optimizer)
    test(epoch, model, testloader)

    if args.checkpoint_interval > 0:
        if epoch % args.checkpoint_interval == 0:
            p = path.join(args.checkpoint_dir,
                          'capsnet_{}_{}.pth'.format(start, epoch))
            torch.save(model.state_dict(),
                       p)
