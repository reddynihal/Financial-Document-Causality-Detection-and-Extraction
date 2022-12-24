import math

from transformers import BertForTokenClassification
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


class BertForTokenClassificationMultiHead(BertForTokenClassification):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config)

        self.tag_mapping = self.get_params(kwargs, "tag_mapping", {'C': 1, 'E': 2, None: 3})

        self.device = self.get_params(kwargs, "device", "cuda:0")

        self.n_heads = self.get_params(kwargs, "n_heads", 1)
        self.n_left_shifted_heads = self.get_params(kwargs, "n_left_shifted_heads", 0)
        self.n_right_shifted_heads = self.get_params(kwargs, "n_right_shifted_heads", 0)

        # Cause heads. Contains only the causes and the effects changed to empty token.
        self.n_cause_heads = self.get_params(kwargs, "n_cause_heads", 0)

        # Effect heads. Contains only the effects and the causes changed to empty token.
        self.n_effect_heads = self.get_params(kwargs, "n_effect_heads", 0)

        self.n_all_heads = self.n_heads + self.n_left_shifted_heads + self.n_right_shifted_heads + \
                           self.n_cause_heads + self.n_effect_heads

        # Handling dropout parameters.
        # The number of dropout parameters should be the same as the number of heads. If length of dropout probs is 1,
        # we generate a prob for all head with that value.
        self.dropout_prob = self.get_params(kwargs, "dropout_prob", [0.0])

        if len(self.dropout_prob) == 1:
            self.dropout_prob = self.dropout_prob * self.n_all_heads

        assert len(self.dropout_prob) == self.n_all_heads, \
            "The number of dropout parameters should be the same as the number of heads." \
            "Please check the --dropout_prob parameter."

        # dropout after classification on a head
        self.after_dropout_prob = self.get_params(kwargs, "after_dropout_prob", [0.0])

        if len(self.after_dropout_prob) == 1:
            self.after_dropout_prob = self.after_dropout_prob * self.n_all_heads

        assert len(self.after_dropout_prob) == self.n_all_heads, \
            "The number of dropout parameters should be the same as the number of heads." \
            "Please check the --after_dropout_prob parameter."

        self.loss_type = self.get_params(kwargs, "loss_type", "standard")

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
                                    for i in range(self.n_all_heads)])

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
        sequence_output = outputs[0]
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

            # loss of left shifted heads
            if self.n_left_shifted_heads > 0:
                left_shifted_labels = self.convert_labels_left_shift(labels)
                left_shifted_attention_mask = self.convert_attention_mask_left_shift(attention_mask)

            for i in range(self.n_left_shifted_heads):
                global_i = self.n_heads + i
                loss += self.calc_loss(logits_arr[global_i], left_shifted_attention_mask, left_shifted_labels)

            # loss of right shifted heads
            if self.n_right_shifted_heads > 0:
                right_shifted_labels = self.convert_labels_right_shift(labels)
                right_shifted_attention_mask = self.convert_attention_mask_right_shift(attention_mask)

            for i in range(self.n_right_shifted_heads):
                global_i = self.n_heads + self.n_left_shifted_heads + i
                loss += self.calc_loss(logits_arr[global_i], right_shifted_attention_mask, right_shifted_labels)

            # loss of just effect heads
            if self.n_effect_heads > 0:
                effect_labels = self.convert_labels_only_effect(labels)

            for i in range(self.n_effect_heads):
                global_i = self.n_heads + self.n_left_shifted_heads + self.n_right_shifted_heads + i
                loss += self.calc_loss(logits_arr[global_i], attention_mask, effect_labels)

            # loss of just cause heads
            if self.n_cause_heads > 0:
                cause_labels = self.convert_labels_only_cause(labels)

            for i in range(self.n_cause_heads):
                global_i = self.n_heads + self.n_left_shifted_heads + \
                           self.n_right_shifted_heads + self.n_effect_heads + i
                loss += self.calc_loss(logits_arr[global_i], attention_mask, cause_labels)

            #outputs = (loss,) + outputs
            outputs = (loss / self.n_all_heads,) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)


    def calc_loss(self, logits, attention_mask, labels):
        loss_fct = nn.CrossEntropyLoss()

        if self.loss_type == "change_count":
            loss_fct = BertForTokenClassificationMultiHead.change_count_loss
        elif self.loss_type == "change_count_square":
            loss_fct = BertForTokenClassificationMultiHead.change_count_square_loss
        elif self.loss_type == "change_count_square_rev":
            loss_fct = BertForTokenClassificationMultiHead.change_count_square_rev_loss

        if attention_mask is not None:
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, self.num_labels)[active_loss]
            active_labels = labels.view(-1)[active_loss]
            return loss_fct(active_logits, active_labels)
        else:
            return loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

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

    @staticmethod
    def change_count_loss(active_logits, active_labels):
        loss_fct = nn.CrossEntropyLoss()

        pred_changes = BertForTokenClassificationMultiHead.changes_on_labels(active_logits.max(1).indices)
        gold_changes = BertForTokenClassificationMultiHead.changes_on_labels(active_labels)

        change_loss = 0

        if gold_changes < pred_changes:
            change_loss = math.exp((pred_changes - gold_changes)/gold_changes) - 1

        return loss_fct(active_logits, active_labels) + change_loss

    @staticmethod
    def change_count_square_loss(active_logits, active_labels):
        loss_fct = nn.CrossEntropyLoss()

        pred_changes = BertForTokenClassificationMultiHead.changes_on_labels(active_logits.max(1).indices)
        gold_changes = BertForTokenClassificationMultiHead.changes_on_labels(active_labels)

        change_loss = 0

        if gold_changes < pred_changes:
            change_loss = (pred_changes - gold_changes) ** 2

        return loss_fct(active_logits, active_labels) + change_loss

    @staticmethod
    def change_count_square_rev_loss(active_logits, active_labels):
        loss_fct = nn.CrossEntropyLoss()

        # the prediction and the gold labels are swapped
        gold_changes = BertForTokenClassificationMultiHead.changes_on_labels(active_logits.max(1).indices)
        pred_changes = BertForTokenClassificationMultiHead.changes_on_labels(active_labels)

        change_loss = 0

        if (gold_changes < pred_changes) and (gold_changes > 0):
            change_loss = (pred_changes - gold_changes) ** 2

        return loss_fct(active_logits, active_labels) + change_loss

    @staticmethod
    def changes_on_labels(labels):
        return sum([1 for i in range(1, len(labels)) if labels[i] != labels[i-1]])

    @staticmethod
    def convert_labels_left_shift(labels):
        labels = torch.roll(labels, 1, 1)
        for i in range(len(labels)):
            labels[i][0] = 0

        return labels

    @staticmethod
    def convert_attention_mask_left_shift(attention_mask):
        attention_mask = torch.roll(attention_mask, 1, 1)
        for i in range(len(attention_mask)):
            attention_mask[i][0] = 0

        return attention_mask

    @staticmethod
    def convert_labels_right_shift(labels):
        labels = torch.roll(labels, -1, 1)
        for i in range(len(labels)):
            labels[i][len(labels[i]) - 1] = 0

        return labels

    @staticmethod
    def convert_attention_mask_right_shift(attention_mask):
        attention_mask = torch.roll(attention_mask, -1, 1)
        for i in range(len(attention_mask)):
            attention_mask[i][len(attention_mask[i]) - 1] = 0

        return attention_mask

    def convert_labels_only_cause(self, labels):
        return labels.detach().cpu().apply_(lambda x: x if x == self.tag_mapping["C"] else self.tag_mapping[None]).to(self.device)

    def convert_labels_only_effect(self, labels):
        return labels.detach().cpu().apply_(lambda x: x if x == self.tag_mapping["E"] else self.tag_mapping[None]).to(self.device)