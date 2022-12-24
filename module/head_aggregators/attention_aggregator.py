import torch
from torch import nn


class AttentionAggregator(nn.Module):

    def __init__(self, num_labels, n_heads, device):
        super(AttentionAggregator, self).__init__()

        self.weights = torch.nn.Parameter(torch.rand(num_labels, n_heads, requires_grad=True).to(device), requires_grad=True)

    def forward(self, x):
        #return self.aggregator(torch.cat(x, 2))

        return (torch.stack(x, 3) *
                torch.stack(
                    [torch.stack(
                        [torch.nn.functional.softmax(self.weights)] * x[0].shape[1]
                    )] * x[0].shape[0]
                )).sum(3)
