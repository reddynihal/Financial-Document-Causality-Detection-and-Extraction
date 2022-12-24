import numpy as np
import pandas as pd
import torch
import transformers
from torch.utils.data import TensorDataset


class CSVTensorDatasetMultiSequence(torch.utils.data.TensorDataset):
    r"""Dataset wrapping tensors for csv files.
    The csv should contains two column: text, label

    Arguments:
        tokenizer_name: transformers tokenizer class name
        max_len: maximum length of texts
    """

    def __init__(self, csv_file_name, tokenizer_name="bert-base-uncased", max_len=510, sequence_num=1):
        data = pd.read_csv(csv_file_name).fillna("")

        tokenizer_class = transformers.BertTokenizer.from_pretrained(tokenizer_name)
        tokenizer = tokenizer_class.from_pretrained(tokenizer_name)

        tokenized_data = data.text.apply((lambda x: tokenizer.encode(x,
                                                                     add_special_tokens=False,
                                                                     max_length=(max_len - 2) * sequence_num,
                                                                     return_overflowing_tokens=True,
                                                                     pad_to_max_length=True)))

        padded_data = np.array([[
            [tokenizer.cls_token_id] + list(part) + [tokenizer.sep_token_id]
            for part in np.array_split(doc, sequence_num)]
            for doc in tokenized_data
        ])

        #padded_data = np.array([i[0:max_len] + [0] * (max_len - len(i[0:max_len])) for i in tokenized_data.values])

        attention_mask = np.where(padded_data != 0, 1, 0)

        inputs = torch.tensor(padded_data)

        masks = torch.tensor(attention_mask)

        labels = torch.tensor(data.label.values)

        self.num_labels = len(data.label.unique())

        super().__init__(inputs, masks, labels)