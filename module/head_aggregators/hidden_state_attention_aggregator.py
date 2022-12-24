import torch
from torch import nn


class HiddenStateAttentionAggregator(nn.Module):

    def __init__(self, num_labels, n_heads, hidden_size, device):
        super(HiddenStateAttentionAggregator, self).__init__()
        self.n_heads = n_heads
        self.num_labels = num_labels

        self.weights = torch.nn.Parameter(torch.rand(hidden_size, n_heads, requires_grad=True).to(device), requires_grad=True)

    def forward(self, x, output):
        mul = torch.matmul(output, self.weights)

        return torch.matmul(mul.unsqueeze(2), torch.stack(x, 2)).squeeze(2)

        # y = [torch.matmul(x[i].unsqueeze(3), mul[:,:,i].unsqueeze(2).unsqueeze(2)).squeeze(3) for i in range(len(x))]
        # return torch.sum(torch.stack(y), 0)

