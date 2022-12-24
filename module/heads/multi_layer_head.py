import math

from torch import nn


class MultiLayerHead(nn.Module):

    def __init__(self, hidden_size, num_labels, num_layers, dropout_prob=0.1, after_dropout_prob=0.0):
        super(MultiLayerHead, self).__init__()

        self.dropout = nn.Dropout(dropout_prob)

        self.linear1 = nn.Linear(hidden_size * num_layers, math.floor(hidden_size * num_layers / 2))

        self.linear2 = nn.Linear(math.floor(hidden_size * num_layers / 2), num_labels)

        self.act_f = nn.ReLU()

        self.dropout_after = nn.Dropout(after_dropout_prob)

    def forward(self, x):
        x = self.dropout(x)
        x = self.linear1(x)
        x = self.act_f(x)
        x = self.linear2(x)
        x = self.act_f(x)
        x = self.dropout_after(x)

        return x
