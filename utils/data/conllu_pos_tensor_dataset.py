import numpy as np
import torch
import transformers
import pyconll

from torch.utils.data import TensorDataset


class ConlluPosTensorDataset(torch.utils.data.TensorDataset):
    r"""Dataset wrapping tensors for cunllu files.

    Arguments:
        tokenizer_name: transformers tokenizer class name
        max_len: maximum length of texts
    """

    def __init__(self, file_name, tokenizer_name="bert-base-uncased", max_len=510, tag_mapping=dict()):
        self.tag_mapping = tag_mapping

        tokens, tags = self.read_sequences(file_name)

        tokenizer_class = transformers.BertTokenizer.from_pretrained(tokenizer_name)
        tokenizer = tokenizer_class.from_pretrained(tokenizer_name)

        # split tokens to bert word pieces
        tokens, tags = self.split_tokens_and_tags_2_word_pieces(tokens, tags, tokenizer)

        # add special tokens, pad and trim to the sentences
        tokens = [self.handle_sequence(seq, max_len, tokenizer) for seq in tokens]
        tags = [self.handle_sequence(seq, max_len, tokenizer) for seq in tags]

        # convert tokens to bert ids
        token_ids = np.array([tokenizer.convert_tokens_to_ids(seq) for seq in tokens])

        # convert tags to integers
        tag_ids = self.convert_tags_to_ids(tags)

        attention_mask = np.where(token_ids != 0, 1, 0)

        inputs = torch.tensor(token_ids)

        masks = torch.tensor(attention_mask)

        labels = torch.tensor(tag_ids)

        super().__init__(inputs, masks, labels)

    @staticmethod
    def read_sequences(file_name):
        data = pyconll.load_from_file(file_name)

        tokens = [[token.form for token in sent] for sent in data]
        tags = [[token.upos for token in sent] for sent in data]

        return [tokens, tags]

    def handle_sequence(self, seq, max_len, tokenizer):
        return self.handle_sequence_by_tokens(seq, max_len, tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token)

    def handle_sequence_by_tokens(self, seq, max_len, start_token, sep_token, padding_token):
        seq = self.resize(seq, max_len-2)
        seq = self.add_start(seq, start_token)
        seq = self.add_sep(seq, sep_token)
        seq = self.padding(seq, max_len, padding_token)
        return seq

    def resize(self, seq, max_len):
        return seq[:max_len]

    def add_start(self, seq, token):
        return [token] + seq

    def add_sep(self, seq, token):
        return seq + [token]

    def padding(self, seq, max_len, token):
        return seq + [token] * (max_len - len(seq))

    def convert_tags_to_ids(self, tags):
        next_id = 0

        for seq in tags:
            for tag in seq:
                if tag not in self.tag_mapping:
                    self.tag_mapping[tag] = next_id
                    next_id += 1

        ids = [[self.tag_mapping[tag] for tag in seq] for seq in tags]

        return ids

    def split_tokens_and_tags_2_word_pieces(self, tokens, tags, tokenizer):
        tokens_wp = []
        tags_wp = []

        for token_seq, tag_seq in zip(tokens, tags):
            token_wp_seq = []
            tag_wp_seq = []
            tokens_wp.append(token_wp_seq)
            tags_wp.append(tag_wp_seq)

            for token, tag in zip(token_seq, tag_seq):
                word_pieces = tokenizer.tokenize(token)

                if (word_pieces is None) or (len(word_pieces) == 0):
                    word_pieces = [tokenizer.unk_token]

                for word_piece in word_pieces:
                    token_wp_seq.append(word_piece)
                    tag_wp_seq.append(tag)

        return tokens_wp, tags_wp
