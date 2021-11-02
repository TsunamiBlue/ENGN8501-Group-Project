import torch
import torch.nn.functional as F
import time

from torch.nn.modules.loss import _WeightedLoss


def get_accuracy(output, type='real'):
    assert type in ['real', 'refine']

    score = F.softmax(output, dim=1)
    class1 = score.cpu().data.numpy()[:, 0]
    class2 = score.cpu().data.numpy()[:, 1]

    if type == 'real':
        return (class1 < class2).mean()
    else:
        return (class1 > class2).mean()


def loop_iter(dataloader):
    while True:
        for data in iter(dataloader):
            yield data


class LocalAdversarialLoss(_WeightedLoss):
    def __init__(self, weight=None, size_average=False, batch_size=128):
        super(LocalAdversarialLoss, self).__init__(weight, size_average)
        self.batch_size = batch_size

    def forward(self, input, target):
        loss = F.cross_entropy(input, target, reduce=False)
        # loss = loss.view(self.batch_size, -1)
        loss = torch.sum(loss) / input.size()[0]  # self.batch_size
        return loss


class MyTimer():
    def __init__(self):
        self.time_dict = {}
        self.count_dict = {}
        self.t0 = 0.

    def track(self):
        self.t0 = time.time()

    def add_value(self, title):
        value = time.time() - self.t0
        self.t0 = 0.

        if not title in self.time_dict:
            self.time_dict[title] = value
            self.count_dict[title] = 1
        else:
            self.time_dict[title] = (self.time_dict[title] * self.count_dict[title] + value)
            self.count_dict[title] += 1
            self.time_dict[title] /= self.count_dict[title]

    def get_all_time(self):
        return self.time_dict