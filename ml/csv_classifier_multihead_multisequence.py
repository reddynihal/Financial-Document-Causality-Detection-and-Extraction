from ml.csv_classifier_multihead import CSVClassifierMultiHead

from torch.utils.data import DataLoader, SequentialSampler, RandomSampler

from module.bert_for_sequence_classification_multi_head_multi_sequence import \
    BertForSequenceClassificationMultiHeadMultiSequence
from utils.data.csv_tensor_dataset_multi_sequence import CSVTensorDatasetMultiSequence

class CSVClassifierMultiHeadMultiSequence(CSVClassifierMultiHead):

    def load_data(self, file, sampler=SequentialSampler):
        data = CSVTensorDatasetMultiSequence(file, max_len=self.config.max_len, tokenizer_name=self.config.model_name,
                                             sequence_num=self.config.sequence_num)
        sampler = sampler(data)
        data_loader = DataLoader(data, sampler=sampler, batch_size=self.config.batch_size, num_workers=0)

        if self.num_labels == 0:
            self.num_labels = data.num_labels

        return data_loader

    def load_model(self, num_labels=2):
        model = BertForSequenceClassificationMultiHeadMultiSequence.from_pretrained(
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
            aggregation_type=self.config.aggregation_type,
            sequence_num=self.config.sequence_num
        )

        # Tell pytorch to run this model on the GPU.
        model.to(self.config.device)

        return model