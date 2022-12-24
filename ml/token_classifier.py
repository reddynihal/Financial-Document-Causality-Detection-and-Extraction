import numpy as np
import pyconll

from torch.utils.data import DataLoader, SequentialSampler

from transformers import BertForTokenClassification, BertTokenizer

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from ml.csv_classifier import CSVClassifier
from utils.data.conllu_pos_tensor_dataset import ConlluPosTensorDataset


class TokenClassifier(CSVClassifier):

    def __init__(self, config):
        self.config = config
        self.tag_mapping = dict()
        self.tag_mapping_inverted = dict()

        self.num_labels = 0

    def train(self):
        train_data_loader = self.load_data(self.config.train_file, is_train_data=True)  # , RandomSampler)
        validation_data_loader = self.load_data(self.config.validation_file, is_train_data=False)

        self.train_on_dataset(train_data_loader, validation_data_loader)

    def load_model(self, num_labels=2):
        model = BertForTokenClassification.from_pretrained(
            self.config.model_name,
            output_attentions=False,
            output_hidden_states=False,
            num_labels=len(self.tag_mapping)
        )

        model.to(self.config.device)

        return model

    def load_data(self, file, sampler=SequentialSampler, is_train_data=True):
        if is_train_data:
            data = ConlluPosTensorDataset(
                file,
                max_len=self.config.max_len,
                tokenizer_name=self.config.model_name
            )

            self.tag_mapping = data.tag_mapping
            self.tag_mapping_inverted = {v: k for k, v in self.tag_mapping.items()}

        else:
            data = ConlluPosTensorDataset(
                file,
                max_len=self.config.max_len,
                tokenizer_name=self.config.model_name,
                tag_mapping=self.tag_mapping
            )

            self.tag_mapping = data.tag_mapping
            self.tag_mapping_inverted = {v: k for k, v in self.tag_mapping.items()}

        sampler = sampler(data)
        data_loader = DataLoader(data, sampler=sampler, batch_size=self.config.batch_size, num_workers=0)

        return data_loader

    def evaluate(self, input_id_list, label_list, logit_list):
        preds = np.argmax(logit_list, axis=2)
        self.evaluate_head(label_list, preds)

    def evaluate_head(self, y_true, y_pred):
        """
        Evaluate Precision, Recall, F1 scores between y_true and y_pred
        If output_file is provided, scores are saved in this file otherwise printed to std output.
        :param y_true: true labels
        :param y_pred: predicted labels
        :return: list of scores (F1, Recall, Precision, ExactMatch)
        """

        y_true_filltered = []
        y_pred_filltered = []

        irrelevant_tokens = ["[CLS]", "[SEP]", "[PAD]"]
        irrelevant_token_ids = [self.tag_mapping[token] for token in irrelevant_tokens]

        for y_t, y_p in zip(np.array(y_true).flatten(), y_pred.flatten()):
            if y_t not in irrelevant_token_ids:
                y_true_filltered.append(y_t)
                y_pred_filltered.append(y_p)

        assert len(y_true) == len(y_pred)

        print("Accuracy: " + str(accuracy_score(y_true_filltered, y_pred_filltered)))
        precision, recall, f1, _ = precision_recall_fscore_support(y_true_filltered, y_pred_filltered, labels=[0, 1],
                                                                   average='weighted')
        scores = [
            "F1: %f\n" % f1,
            "Recall: %f\n" % recall,
            "Precision: %f\n" % precision,
            "ExactMatch: %f\n" % -1.0
        ]
        for s in scores:
            print(s, end='')

    def write_results(self, input_id_list, label_list, logit_list):
        self.write_results_on_head(input_id_list, label_list, logit_list)

    def write_results_on_head(self, input_id_list, label_list, logit_list, file_name_params=""):
        if self.config.out_file is not None:
            conll_data = pyconll.load_from_file(self.config.validation_file)
            predicted_label_ids = np.argmax(logit_list, axis=2)

            tokenizer_class = BertTokenizer.from_pretrained(self.config.model_name)
            tokenizer = tokenizer_class.from_pretrained(self.config.model_name)

            word_piece_list = [tokenizer.convert_ids_to_tokens(input_ids) for input_ids in input_id_list]
            original_label_list = self.convert_label_ids_2_labels_2d(label_list)
            predicted_label_list = self.convert_label_ids_2_labels_2d(predicted_label_ids)

            for i in range(len(conll_data)):
                sentence_offset = 1

                for j in range(len(conll_data[i])):

                    # set the pos tag
                    if j + sentence_offset < len(predicted_label_list[i]) - 1:
                        conll_data[i][j].upos = predicted_label_list[i][j + sentence_offset]
                    else:
                        conll_data[i][j].upos = "_"

                    # increase the offset, if the number of word pieces are larger than 1
                    word_pieces = tokenizer.tokenize(conll_data[i][j].form)

                    if j + sentence_offset < len(word_piece_list[i]) and word_pieces[0] != word_piece_list[i][j + sentence_offset]:
                        print("sentence align error: " + word_pieces[0] + " " + word_piece_list[i][j + sentence_offset])

                    if len(word_pieces) > 1:
                        sentence_offset += len(word_pieces) - 1

                    if len(word_pieces) == 0:
                        print("the length of word pieces is 0 for the token: " + conll_data[i][j].form)

            conll_data.write(open(self.config.out_file + str(self.train_epoch) + "." + file_name_params + ".conllu", "w"))

    @staticmethod
    def is_non_starting_word_piece(word_piece):
        return word_piece[:2] != "##"

    @staticmethod
    def is_extended_label(label, tokenizer):
        return tokenizer.cls_token == label or tokenizer.sep_token == label or tokenizer.pad_token == label

    def convert_label_ids_2_labels_1d(self, labels_1d):
        return [self.tag_mapping_inverted[label] for label in labels_1d]

    def convert_label_ids_2_labels_2d(self, labels_2d):
        return [self.convert_label_ids_2_labels_1d(labels_1d) for labels_1d in labels_2d]