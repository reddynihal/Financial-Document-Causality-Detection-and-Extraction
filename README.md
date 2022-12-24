# Multi-headed transformer based approach to Causality Detection and Extraction in Financial Text

### Token classifier

##### Input
conll format
Labels in the 4th column

##### Parameters
```
usage: train_token_classification_cli.py [-h] -t TRAIN_FILE -v VALIDATION_FILE
                                         [-d DEVICE] [-b BATCH_SIZE]
                                         [-e EPOCHS] [-m MODEL_OUT]
                                         [-c CLASSIFIER] [-s SEED]
                                         [-l MAX_LEN] [-n MODEL_NAME]
                                         [-f N_HEADS] [-o OUT_FILE]
                                         [--lt LOSS_TYPE]
                                         [--n_left_shifted_heads N_LEFT_SHIFTED_HEADS]
                                         [--n_right_shifted_heads N_RIGHT_SHIFTED_HEADS]
                                         [--n_cause_heads N_CAUSE_HEADS]
                                         [--n_effect_heads N_EFFECT_HEADS]
                                         [--dropout_prob DROPOUT_PROB [DROPOUT_PROB ...]]
                                         [--after_dropout_prob AFTER_DROPOUT_PROB [AFTER_DROPOUT_PROB ...]]
                                         [--selected_layer SELECTED_LAYERS [SELECTED_LAYERS ...]]
                                         [--selected_layers_by_heads SELECTED_LAYERS_BY_HEADS [SELECTED_LAYERS_BY_HEADS ...]]
                                         [--head_type HEAD_TYPE]
                                         [--aggregation_type AGGREGATION_TYPE]

optional arguments:
  -h, --help            show this help message and exit
  -t TRAIN_FILE, --train TRAIN_FILE
                        path of the train file
  -v VALIDATION_FILE, --validation VALIDATION_FILE
                        path of the validation file
  -d DEVICE, --device DEVICE
                        device_name
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        batch size
  -e EPOCHS, --epochs EPOCHS
                        number of epochs
  -m MODEL_OUT, --model_out MODEL_OUT
                        model saving path
  -c CLASSIFIER, --classifier CLASSIFIER
                        classifier type name [singlehead, multihead]
  -s SEED, --seed SEED  seed
  -l MAX_LEN, --max_len MAX_LEN
                        max length of tokenized instances
  -n MODEL_NAME, --model_name MODEL_NAME
                        name of bert model
  -f N_HEADS, --n_heads N_HEADS
                        number of heads
  -o OUT_FILE, --out_file OUT_FILE
                        output file path
  --lt LOSS_TYPE, --loss_type LOSS_TYPE
                        lost function: standard|change_count|change_count_squa
                        re|change_count_square_rev
  --n_left_shifted_heads N_LEFT_SHIFTED_HEADS
                        number of left shifted heads
  --n_right_shifted_heads N_RIGHT_SHIFTED_HEADS
                        number of right shifted heads
  --n_cause_heads N_CAUSE_HEADS
                        number of cause heads
  --n_effect_heads N_EFFECT_HEADS
                        number of effect heads
  --dropout_prob DROPOUT_PROB [DROPOUT_PROB ...]
                        Dropout probability of the heads. You can specify
                        independent dropout probability for each head. The
                        number of parameters must be one or equal with the
                        number of heads. If the number of parameters is one,
                        all head will use that value.E.g.: --dropout_prob 0.3
                        --dropout_prob 0.1 0.3 0.5 0.6 0.7 # if the number of
                        heads is 5
  --after_dropout_prob AFTER_DROPOUT_PROB [AFTER_DROPOUT_PROB ...]
                        same as in dropout_probs in the top of the heads
  --selected_layer SELECTED_LAYERS [SELECTED_LAYERS ...], --sl SELECTED_LAYERS [SELECTED_LAYERS ...]
                        use the selected layers of bert.Layer 0: embedding
                        layer, Layer 12: output of the bert (base).Each head
                        will get the concatenation of the selected
                        layers.Cannot use with
                        --selected_layer_by_heads.Usage: --sl 5 8 10
  --selected_layers_by_heads SELECTED_LAYERS_BY_HEADS [SELECTED_LAYERS_BY_HEADS ...], --slh SELECTED_LAYERS_BY_HEADS [SELECTED_LAYERS_BY_HEADS ...]
                        use the selected layers of bert.Layer 0: embedding
                        layer, Layer 12: output of the bert (base). The number
                        of layers should be the same as the number of heads.
                        You can add concatenated layers by using the "_"
                        character.Cannot use with --selected_layer.Usage:
                        --slh 6 6 8 12 6_12
  --head_type HEAD_TYPE
                        type of head: base/base_shared/base_relu/multi_layer
  --aggregation_type AGGREGATION_TYPE
                        The method of the aggregation of different heads.If
                        the sum_independent (default) selected the losses are
                        calculated independently on each head. In every other
                        cases the loss is calculated after on the aggregated
                        data.Possible values: sum_independent/sum/linear/atten
                        tion/hidden_state_attention
```

### Sequence classifier

##### Input

Csv with at least two column, *text* and *label*.

##### Parameters
```
usage: train_sequence_classification_cli.py [-h] -t TRAIN_FILE -v
                                            VALIDATION_FILE [-d DEVICE]
                                            [-b BATCH_SIZE] [-e EPOCHS]
                                            [-m MODEL_OUT] [-c CLASSIFIER]
                                            [-s SEED] [-l MAX_LEN]
                                            [-n MODEL_NAME] [-f N_HEADS]
                                            [-o OUT_FILE]
                                            [--dropout_prob DROPOUT_PROB [DROPOUT_PROB ...]]
                                            [--after_dropout_prob AFTER_DROPOUT_PROB [AFTER_DROPOUT_PROB ...]]
                                            [--selected_layer SELECTED_LAYERS [SELECTED_LAYERS ...]]
                                            [--selected_layers_by_heads SELECTED_LAYERS_BY_HEADS [SELECTED_LAYERS_BY_HEADS ...]]
                                            [--head_type HEAD_TYPE]
                                            [--aggregation_type AGGREGATION_TYPE]
                                            [--sequence_num SEQUENCE_NUM]

optional arguments:
  -h, --help            show this help message and exit
  -t TRAIN_FILE, --train TRAIN_FILE
                        path of the train file
  -v VALIDATION_FILE, --validation VALIDATION_FILE
                        path of the validation file
  -d DEVICE, --device DEVICE
                        device_name
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        batch size
  -e EPOCHS, --epochs EPOCHS
                        number of epochs
  -m MODEL_OUT, --model_out MODEL_OUT
                        model saving path
  -c CLASSIFIER, --classifier CLASSIFIER
                        classifier type name [singlehead, multihead]
  -s SEED, --seed SEED  seed
  -l MAX_LEN, --max_len MAX_LEN
                        max length of tokenized instances
  -n MODEL_NAME, --model_name MODEL_NAME
                        name of bert model
  -f N_HEADS, --n_heads N_HEADS
                        number of heads
  -o OUT_FILE, --out_file OUT_FILE
                        output file path
  --dropout_prob DROPOUT_PROB [DROPOUT_PROB ...]
                        Dropout probability of the heads. You can specify
                        independent dropout probability for each head. The
                        number of parameters must be one or equal with the
                        number of heads. If the number of parameters is one,
                        all head will use that value.E.g.: --dropout_prob 0.3
                        --dropout_prob 0.1 0.3 0.5 0.6 0.7 # if the number of
                        heads is 5
  --after_dropout_prob AFTER_DROPOUT_PROB [AFTER_DROPOUT_PROB ...]
                        same as in dropout_probs in the top of the heads
  --selected_layer SELECTED_LAYERS [SELECTED_LAYERS ...], --sl SELECTED_LAYERS [SELECTED_LAYERS ...]
                        use the selected layers of bert.Layer 0: embedding
                        layer, Layer 12: output of the bert (base).Each head
                        will get the concatenation of the selected
                        layers.Cannot use with
                        --selected_layer_by_heads.Usage: --sl 5 8 10
  --selected_layers_by_heads SELECTED_LAYERS_BY_HEADS [SELECTED_LAYERS_BY_HEADS ...], --slh SELECTED_LAYERS_BY_HEADS [SELECTED_LAYERS_BY_HEADS ...]
                        use the selected layers of bert.Layer 0: embedding
                        layer, Layer 12: output of the bert (base). The number
                        of layers should be the same as the number of heads.
                        You can add concatenated layers by using the "_"
                        character.Cannot use with --selected_layer.Usage:
                        --slh 6 6 8 12 6_12
  --head_type HEAD_TYPE
                        type of head: base/base_shared/base_relu/multi_layer
  --aggregation_type AGGREGATION_TYPE
                        The method of the aggregation of different heads.If
                        the sum_independent (default) selected the losses are
                        calculated independently on each head. In every other
                        cases the loss is calculated after on the aggregated
                        data.Possible values: sum_independent/sum/linear/atten
                        tion/hidden_state_attention
  --sequence_num SEQUENCE_NUM
                        number of sequences in a document

```
