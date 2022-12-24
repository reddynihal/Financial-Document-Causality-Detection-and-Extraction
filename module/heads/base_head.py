from torch import nn


class BaseHead(nn.Module):

    def __init__(self, hidden_size, num_labels, num_layers, dropout_prob=0.1, after_dropout_prob=0.0):
        super(BaseHead, self).__init__()

        self.dropout = nn.Dropout(dropout_prob)

        self.classifier = nn.Linear(hidden_size * num_layers, num_labels)

        self.dropout_after = nn.Dropout(after_dropout_prob)

    def forward(self, x):
        return self.dropout_after(self.classifier(self.dropout(x)))
