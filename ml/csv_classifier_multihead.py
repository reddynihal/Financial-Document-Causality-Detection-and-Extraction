import numpy as np

from module.bert_for_sequence_classification_multi_head import BertForSequenceClassificationMultiHead
from ml.csv_classifier import CSVClassifier


class CSVClassifierMultiHead(CSVClassifier):

    def load_model(self, num_labels=2):
        model = BertForSequenceClassificationMultiHead.from_pretrained(
            self.config.model_name,  # Use the 12-layer BERT model, with an uncased vocab.
            num_labels=num_labels,  # The number of output labels--2 for binary classification.
            # You can increase this for multi-class tasks.
            output_attentions=False,  # Whether the model returns attentions weights.
            output_hidden_states=False,  # Whether the model returns all hidden-states.
            n_heads=self.config.n_heads,
            hidden_dropout_prob=self.config.dropout_prob[0],
            dropout_prob=self.config.dropout_prob,
            after_dropout_prob=self.config.after_dropout_prob,
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
            logit_list += list(np.array([i.detach().cpu().numpy() for i in b_output[1]]).transpose(1, 0, 2))

        return logit_list

    def evaluate(self, input_ids_list, labels_list, logits_list):
        for head in range(self.config.n_heads):
            logits = [i[head] for i in logits_list]
            preds = np.argmax(logits, axis=1).flatten()
            print()
            print("Head " + str(head) + " ====================")

            self.evaluate_head(labels_list, preds)

        # sum of all model
        logits = [sum(i) for i in logits_list]
        preds = np.argmax(logits, axis=1).flatten()
        print()
        print("Aggregated head ====================")

        self.evaluate_head(labels_list, preds)

    def write_results(self, input_id_list, label_list, logit_list):
        for head in range(self.config.n_heads):
            logits = [i[head] for i in logit_list]
            self.write_results_on_head(input_id_list, label_list, logits, "h_" + str(head))

        logits = [i[-1] for i in logit_list]
        self.write_results_on_head(input_id_list, label_list, logits, "h_sum")