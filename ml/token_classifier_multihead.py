import numpy as np

from module.bert_for_token_classification_multi_head import BertForTokenClassificationMultiHead
from ml.token_classifier import TokenClassifier


class TokenClassifierMultiHead(TokenClassifier):

    def load_model(self, num_labels=2):
        model = BertForTokenClassificationMultiHead.from_pretrained(
            self.config.model_name,
            output_attentions=False,
            output_hidden_states=True,
            num_labels=len(self.tag_mapping),
            hidden_dropout_prob=self.config.dropout_prob[0],
            dropout_prob=self.config.dropout_prob,
            after_dropout_prob=self.config.after_dropout_prob,
            n_heads=self.config.n_heads,
            loss_type=self.config.loss_type,
            tag_mapping=self.tag_mapping,
            device=self.config.device,
            n_left_shifted_heads=self.config.n_left_shifted_heads,
            n_right_shifted_heads=self.config.n_right_shifted_heads,
            n_cause_heads=self.config.n_cause_heads,
            n_effect_heads=self.config.n_effect_heads,
            selected_layers=self.config.selected_layers,
            selected_layers_by_heads=self.config.selected_layers_by_heads,
            head_type=self.config.head_type,
            aggregation_type=self.config.aggregation_type
        )

        # Tell pytorch to run this model on the GPU.
        model.to(self.config.device)

        return model

    def convert_batch_of_outputs_to_list_of_logits(self, output):
        logit_list = []

        for b_output in output:
            logit_list += list(np.array([i.detach().cpu().numpy() for i in b_output[1]]).transpose(1, 0, 2, 3))

        return logit_list

    def evaluate(self, input_id_list, label_list, logit_list):
        for head in range(self.config.n_heads):
            logits = [i[head] for i in logit_list]
            preds = np.argmax(logits, axis=2)
            print()
            print("Head " + str(head) + " ====================")

            self.evaluate_head(label_list, preds)

        # aggregation of all head
        logits = [i[-1] for i in logit_list]
        preds = np.argmax(logits, axis=2)
        print()
        print("Aggregated head ====================")

        self.evaluate_head(label_list, preds)

    def write_results(self, input_id_list, label_list, logit_list):
        for head in range(self.config.n_heads):
            logits = [i[head] for i in logit_list]
            self.write_results_on_head(input_id_list, label_list, logits, "h_" + str(head))

        logits = [i[-1] for i in logit_list]
        self.write_results_on_head(input_id_list, label_list, logits, "h_sum")
