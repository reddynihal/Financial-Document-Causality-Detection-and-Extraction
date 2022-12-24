import torch
from torch import nn


class LinearAggregator(nn.Module):

    def __init__(self, num_labels, n_heads):
        super(LinearAggregator, self).__init__()

        self.aggregator = nn.Linear(num_labels * n_heads, num_labels)

    def forward(self, x):
        return self.aggregator(torch.cat(x, len(x[0].shape) - 1))
