from torch import nn


class BaseSharedHead(nn.Module):

    def __init__(self, hidden_size, num_labels, num_layers, dropout_prob=0.1, after_dropout_prob=0.0):
        super(BaseSharedHead, self).__init__()

        self.dropout = nn.Dropout(dropout_prob)

        self.classifier = SharedLinearSingleton.get_instance(hidden_size, num_labels, num_layers)

        self.dropout_after = nn.Dropout(after_dropout_prob)

    def forward(self, x):
        return self.dropout_after(self.classifier(self.dropout(x)))


class SharedLinearSingleton:
    __instance = None

    @staticmethod
    def get_instance(hidden_size, num_labels, num_layers):

        if SharedLinearSingleton.__instance is None:
            SharedLinearSingleton(hidden_size, num_labels, num_layers)
        return SharedLinearSingleton.__instance

    def __init__(self, hidden_size, num_labels, num_layers):

        if SharedLinearSingleton.__instance is not None:
            raise Exception("This class is a singleton!")
        else:
            SharedLinearSingleton.__instance = nn.Linear(hidden_size * num_layers, num_labels)
