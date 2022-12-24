from ml.csv_classifier import CSVClassifier
from ml.csv_classifier_multihead import CSVClassifierMultiHead
from ml.csv_classifier_multihead_multisequence import CSVClassifierMultiHeadMultiSequence

from argparse import ArgumentParser


def classify_from_args(args):
    if args.classifier == "singlehead":
        classifier = CSVClassifier(args)
    else:
        if args.sequence_num == 1:
            classifier = CSVClassifierMultiHead(args)
        else:
            classifier = CSVClassifierMultiHeadMultiSequence(args)


    classifier.train()


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("-t", "--train", dest="train_file",
                            help="path of the train file", required=True)

    arg_parser.add_argument("-v", "--validation", dest="validation_file",
                            help="path of the validation file", required=True)

    arg_parser.add_argument("-d", "--device", dest="device",
                            help="device_name", default="cuda")

    arg_parser.add_argument("-b", "--batch_size", dest="batch_size",
                            help="batch size", default="4", type=int)

    arg_parser.add_argument("-e", "--epochs", dest="epochs",
                            help="number of epochs", default="4", type=int)

    arg_parser.add_argument("-m", "--model_out", dest="model_out",
                            help="model saving path", default=None)

    arg_parser.add_argument("-c", "--classifier", dest="classifier",
                            help="classifier type name [singlehead, multihead]", default="multihead")

    arg_parser.add_argument("-s", "--seed", dest="seed",
                            help="seed", default=42, type=int)

    arg_parser.add_argument("-l", "--max_len", dest="max_len",
                            help="max length of tokenized instances", default=510, type=int)

    arg_parser.add_argument("-n", "--model_name", dest="model_name",
                            help="name of bert model", default="bert-base-uncased")

    arg_parser.add_argument("-f", "--n_heads", dest="n_heads",
                            help="number of heads", default=1, type=int)

    arg_parser.add_argument("-o", "--out_file", dest="out_file",
                            help="output file path", default=None)

    arg_parser.add_argument("--dropout_prob", dest="dropout_prob",
                            help="Dropout probability of the heads. "
                                 "You can specify independent dropout probability for each head. "
                                 "The number of parameters must be one or equal with the number of heads. "
                                 "If the number of parameters is one, all head will use that value."
                                 "E.g.: --dropout_prob 0.3 \n"
                                 "--dropout_prob 0.1 0.3 0.5 0.6 0.7 # if the number of heads is 5",
                            nargs='+', default=[0.1], type=float)

    arg_parser.add_argument("--after_dropout_prob", dest="after_dropout_prob",
                            help="same as in dropout_probs in the top of the heads",
                            nargs='+', default=[0.0], type=float)

    arg_parser.add_argument('--selected_layer', "--sl", dest="selected_layers", nargs='+', type=int,
                            help="use the selected layers of bert."
                                 "Layer 0: embedding layer, Layer 12: output of the bert (base)."
                                 "Each head will get the concatenation of the selected layers."
                                 "Cannot use with --selected_layer_by_heads."
                                 "Usage:\n --sl 5 8 10")

    arg_parser.add_argument('--selected_layers_by_heads', "--slh", dest="selected_layers_by_heads", nargs='+', type=int,
                            help="use the selected layers of bert."
                                 "Layer 0: embedding layer, Layer 12: output of the bert (base). "
                                 "The number of layers should be the same as the number of heads. "
                                 "You can add concatenated layers by using the \"_\" character."
                                 "Cannot use with --selected_layer."
                                 "Usage:\n --slh 6 6 8 12 6_12")

    arg_parser.add_argument("--head_type", dest="head_type",
                            help="type of head: base/base_shared/base_relu/multi_layer", default="base")

    arg_parser.add_argument("--aggregation_type", dest="aggregation_type",
                            help="The method of the aggregation of different heads."
                                 "If the sum_independent (default) selected the losses are calculated independently on "
                                 "each head. In every other cases the loss is calculated after on the aggregated data."
                                 "Possible values: sum_independent/sum/linear/attention/hidden_state_attention",
                            default="sum_independent")

    arg_parser.add_argument("--sequence_num", dest="sequence_num",
                            help="number of sequences in a document", type=int, default=1)

    args = arg_parser.parse_args()

    classify_from_args(args)
