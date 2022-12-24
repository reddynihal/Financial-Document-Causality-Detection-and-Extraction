import torch
from torch import nn


class SumAggregator(nn.Module):

    def forward(self, x):
        return torch.sum(torch.stack(x), 0)
