import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F


def compute_stats(X_train, y_train, X_test, y_test, opfun, accfun, ghost_batch=128):
    """
    Computes training loss, test loss, and test accuracy efficiently.

    :param X_train: set of training examples
    :param y_train: set of training labels
    :param X_test: set of test examples
    :param y_test: set of test labels
    :param opfun: computes forward pass over network over sample Sk
    :param accfun: computes accuracy against labels
    :param ghost_batch: maximum size of effective batch (default: 128)
    :return train_loss: training loss
    :return test_loss: test loss
    :return test_acc: test accuracy
    """

    # compute test loss and test accuracy
    test_loss = 0
    test_acc = 0

    # loop through test data
    for smpl in np.array_split(np.random.permutation(range(X_test.shape[0])), int(X_test.shape[0]/ghost_batch)):

        # define test set targets
        if torch.cuda.is_available():
            test_tgts = torch.from_numpy(y_test[smpl]).cuda().long().squeeze()
        else:
            test_tgts = torch.from_numpy(y_test[smpl]).long().squeeze()

        # define test set ops
        testops = opfun(X_test[smpl])

        # accumulate weighted test loss and test accuracy
        if torch.cuda.is_available():
            test_loss += F.cross_entropy(testops, test_tgts).cpu().item()*(len(smpl)/X_test.shape[0])
        else:
            test_loss += F.cross_entropy(testops, test_tgts).item()*(len(smpl)/X_test.shape[0])

        test_acc += accfun(testops, y_test[smpl])*(len(smpl)/X_test.shape[0])

    # compute training loss
    train_loss = 0

    # loop through training data
    for smpl in np.array_split(np.random.permutation(range(X_train.shape[0])), int(X_test.shape[0]/ghost_batch)):

        # define training set targets
        if torch.cuda.is_available():
            train_tgts = torch.from_numpy(y_train[smpl]).cuda().long().squeeze()
        else:
            train_tgts = torch.from_numpy(y_train[smpl]).long().squeeze()

        # define training set ops
        trainops = opfun(X_train[smpl])

        # accumulate weighted training loss
        if torch.cuda.is_available():
            train_loss += F.cross_entropy(trainops, train_tgts).cpu().item()*(len(smpl)/X_train.shape[0])
        else:
            train_loss += F.cross_entropy(trainops, train_tgts).item()*(len(smpl)/X_train.shape[0])

    return train_loss, test_loss, test_acc


# Compute Objective and Gradient Helper Function

def get_grad(optimizer, X_Sk, y_Sk, opfun, ghost_batch=128, return_=True):
    """
    Computes objective and gradient of neural network over data sample.

    :param optimizer: the PBQN optimizer
    :param X_Sk: set of training examples over sample Sk
    :param y_Sk: set of training labels over sample Sk
    :param opfun: computes forward pass over network over sample Sk
    :param ghost_batch: maximum size of effective batch (default: 128)
    :param return_: util use
    :return grad: stochastic gradient over sample Sk
    :return obj: stochastic function value over sample Sk
    """
    if(torch.cuda.is_available()):
        obj = torch.tensor(0, dtype=torch.float).cuda()
    else:
        obj = torch.tensor(0, dtype=torch.float)

    Sk_size = X_Sk.shape[0]

    optimizer.zero_grad()

    # loop through relevant data
    for idx in np.array_split(np.arange(Sk_size), max(int(Sk_size/ghost_batch), 1)):

        # define ops
        ops = opfun(X_Sk[idx])

        # define targets
        if(torch.cuda.is_available()):
            tgts = Variable(torch.from_numpy(y_Sk[idx]).cuda().long().squeeze())
        else:
            tgts = Variable(torch.from_numpy(y_Sk[idx]).long().squeeze())

        # define loss and perform forward-backward pass
        loss_fn = F.cross_entropy(ops, tgts)*(len(idx)/Sk_size)
        loss_fn.backward()

        # accumulate loss
        obj += loss_fn

    # gather flat gradient
    if return_:
        grad = optimizer._gather_flat_grad()

        return grad, obj


def adjust_learning_rate(optimizer, learning_rate):
    """
    Sets the learning rate of optimizer.

    Implemented by: Hao-Jun Michael Shi and Dheevatsa Mudigere
    Last edited 8/29/18.

    Inputs:
        optimizer (Optimizer): any optimizer
        learning_rate (float): desired steplength

    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate

    return


class CUTEstFunction(torch.autograd.Function):
    """
    Converts CUTEst problem using PyCUTEst to PyTorch function.

    Implemented by: Hao-Jun Michael Shi and Dheevatsa Mudigere
    Last edited 9/21/18.

    """

    @staticmethod
    def forward(ctx, input, problem):
        x = input.clone().detach().numpy()
        obj, grad = problem.obj(x, gradient=True)
        ctx.save_for_backward(torch.tensor(grad, dtype=torch.float))
        return torch.tensor(obj, dtype=torch.float)

    @staticmethod
    def backward(ctx, grad_output):
        grad, = ctx.saved_tensors
        return grad, None


class CUTEstProblem(torch.nn.Module):
    """
    Converts CUTEst problem to torch neural network module.

    Implemented by: Hao-Jun Michael Shi and Dheevatsa Mudigere
    Last edited 9/21/18.

    Inputs:
        problem (callable): CUTEst problem interfaced through PyCUTEst

    """

    def __init__(self, problem):
        super(CUTEstProblem, self).__init__()
        # get initialization
        x = torch.tensor(problem.x0, dtype=torch.float)
        x.requires_grad_()

        # store variables and problem
        self.variables = torch.nn.Parameter(x)
        self.problem = problem

    def forward(self):
        model = CUTEstFunction.apply
        return model(self.variables, self.problem)

    def grad(self):
        return self.variables.grad

    def x(self):
        return self.variables
