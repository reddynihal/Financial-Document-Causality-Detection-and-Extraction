import math

from transformers import BertForSequenceClassification
from transformers.modeling_bert import BertModel
import torch
from torch import nn

from module.head_aggregators.hidden_state_attention_aggregator import HiddenStateAttentionAggregator
from module.heads.base_head import BaseHead
from module.heads.base_relu_head import BaseReluHead
from module.heads.base_shared_head import BaseSharedHead
from module.heads.multi_layer_head import MultiLayerHead
from module.head_aggregators.linear_aggregator import LinearAggregator
from module.head_aggregators.sum_aggregator import SumAggregator
from module.head_aggregators.attention_aggregator import AttentionAggregator


class BertForSequenceClassificationMultiHead(BertForSequenceClassification):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config)

        self.device = self.get_params(kwargs, "device", "cuda:0")

        self.n_heads = self.get_params(kwargs, "n_heads", 1)

        # Handling dropout parameters.
        # The number of dropout parameters should be the same as the number of heads. If length of dropout probs is 1,
        # we generate a prob for all head with that value.
        self.dropout_prob = self.get_params(kwargs, "dropout_prob", [0.0])

        if len(self.dropout_prob) == 1:
            self.dropout_prob = self.dropout_prob * self.n_heads

        assert len(self.dropout_prob) == self.n_heads, \
            "The number of dropout parameters should be the same as the number of heads." \
            "Please check the --dropout_prob parameter."

        # dropout after classification on a head
        self.after_dropout_prob = self.get_params(kwargs, "after_dropout_prob", [0.0])

        if len(self.after_dropout_prob) == 1:
            self.after_dropout_prob = self.after_dropout_prob * self.n_heads

        assert len(self.after_dropout_prob) == self.n_heads, \
            "The number of dropout parameters should be the same as the number of heads." \
            "Please check the --after_dropout_prob parameter."

        self.num_labels = config.num_labels

        self.bert = BertModel(config)

        # select special layers from bert model
        self.selected_layers = self.get_params(kwargs, "selected_layers", None)

        self.num_layers = 1
        if self.selected_layers:
            self.num_layers = len(self.selected_layers)

        self.head_type = self.get_head_type(self.get_params(kwargs, "head_type", "base"))

        self.heads = nn.ModuleList([self.head_type(config.hidden_size, self.config.num_labels, self.num_layers,
                                                   self.dropout_prob[i], self.after_dropout_prob[i])
                                    for i in range(self.n_heads)])

        self.selected_layers_by_heads = self.get_params(kwargs, "selected_layers_by_heads", None)

        assert (not self.selected_layers_by_heads) or (len(self.selected_layers_by_heads) == self.n_all_heads), \
            "The number of selected_layers_by_heads parameters should be the same as the number of heads." \
            "Please check the --selected_layers_by_heads parameter."

        assert not (self.selected_layers_by_heads and self.selected_layers), \
            "You can only use one from the following parameters: selected_layer, selected_layers_by_heads"

        self.aggregation_type = self.get_params(kwargs, "aggregation_type", "sum_independent")

        self.head_aggregator = self.get_aggregation_type(self.aggregation_type, self.num_labels, self.n_heads,
                                                         config.hidden_size, self.device)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        # bert layer selection
        sequence_output = outputs[1]
        if self.selected_layers:
            sequence_output = torch.cat([outputs[2][l] for l in self.selected_layers], 2)  # outputs[2][11]

        if self.selected_layers_by_heads:
            logits_arr = [self.heads[i](outputs[2][i]) for i in range(len(self.heads))]
        else:
            logits_arr = [head(sequence_output) for head in self.heads]

        if self.aggregation_type == "hidden_state_attention":
            # add aggregated heads on case of hidden state attention
            logits_arr += [self.head_aggregator(logits_arr[:self.n_heads], sequence_output)]
        else:
            logits_arr += [self.head_aggregator(logits_arr[:self.n_heads])]  # add aggregated heads

        outputs = (  [0], ) + (logits_arr, ) #+ outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:

            if self.aggregation_type == "sum_independent":
                loss = self.calc_loss(logits_arr[0], attention_mask, labels)

                # loss of standard heads
                for i in range(1, self.n_heads):
                    loss += self.calc_loss(logits_arr[i], attention_mask, labels)
            else:
                # calculate the lost on the aggregated head
                loss = self.calc_loss(logits_arr[-1], attention_mask, labels)

            #outputs = (loss,) + outputs
            outputs = (loss / self.n_heads,) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)

    def calc_loss(self, logits, attention_mask, labels):
        loss_fct = nn.CrossEntropyLoss()
        return loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        #
        # if labels is not None:
        #     loss_fct = nn.CrossEntropyLoss()
        #     loss = loss_fct(logits_arr[0].view(-1, self.num_labels), labels.view(-1))
        #
        #     for i in range(1, self.n_heads):
        #         loss_fct = nn.CrossEntropyLoss()
        #         loss += loss_fct(logits_arr[i].view(-1, self.num_labels), labels.view(-1))
        #
        #     outputs = (loss / self.n_heads,) + outputs
        # if attention_mask is not None:
        #     active_loss = attention_mask.view(-1) == 1
        #     active_logits = logits.view(-1, self.num_labels)[active_loss]
        #     active_labels = labels.view(-1)[active_loss]
        #     return loss_fct(active_logits, active_labels)
        # else:
        #     return loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

    @staticmethod
    def get_params(kwargs, param, default):
        if param in kwargs:
            return kwargs[param]

        return default

    @staticmethod
    def get_head_type(head_type):
        if head_type == "multi_layer":
            return MultiLayerHead
        if head_type == "base_relu":
            return BaseReluHead
        if head_type == "base_shared":
            return BaseSharedHead

        return BaseHead

    @staticmethod
    def get_aggregation_type(aggregation_type, num_labels, n_heads, hidden_size, device):
        if aggregation_type == "linear":
            return LinearAggregator(num_labels, n_heads)
        elif aggregation_type == "attention":
            return AttentionAggregator(num_labels, n_heads, device)
        elif aggregation_type == "hidden_state_attention":
            return HiddenStateAttentionAggregator(num_labels, n_heads, hidden_size, device)

        return SumAggregator()


