import numpy as np
import time
import datetime
import random
import pandas as pd

import torch
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler

from transformers import BertForSequenceClassification, get_linear_schedule_with_warmup, AdamW

from utils.data.csv_tensor_dataset import CSVTensorDataset

from sklearn.metrics import accuracy_score, precision_recall_fscore_support


class CSVClassifier:

    def __init__(self, config):
        self.config = config
        self.train_epoch = 0
        self.num_labels = 0

    def train(self):
        train_data_loader = self.load_data(self.config.train_file, RandomSampler)
        validation_data_loader = self.load_data(self.config.validation_file)

        self.train_on_dataset(train_data_loader, validation_data_loader)

    def validate(self):
        validation_data_loader = self.load_data(self.config.validation_file)

        model = torch.load(self.config.model_input)
        self.validate_on_dataset(model, validation_data_loader)

    def load_model(self, num_labels=7):
        model = BertForSequenceClassification.from_pretrained(
            self.config.model_name,  # Use the 12-layer BERT model, with an uncased vocab.
            num_labels=num_labels,  # The number of output labels--2 for binary classification.
            # You can increase this for multi-class tasks.
            output_attentions=False,  # Whether the model returns attentions weights.
            output_hidden_states=False,  # Whether the model returns all hidden-states.
        )

        # Tell pytorch to run this model on the GPU.
        model.to(self.config.device)

        return model

    def format_time(argsself, elapsed):
        '''
        Takes a time in seconds and returns a string hh:mm:ss
        '''
        # Round to the nearest second.
        elapsed_rounded = int(round((elapsed)))

        # Format as hh:mm:ss
        return str(datetime.timedelta(seconds=elapsed_rounded))

    def train_on_dataset( self, train_dataloader, validation_dataloader):

        model = self.load_model(num_labels=self.num_labels)

        # Note: AdamW is a class from the huggingface library (as opposed to pytorch)
        # I believe the 'W' stands for 'Weight Decay fix"
        optimizer = AdamW(model.parameters(),
                          lr=2e-5,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                          eps=1e-8  # args.adam_epsilon  - default is 1e-8.
                          )

        # Total number of training steps is number of batches * number of epochs.
        total_steps = len(train_dataloader) * self.config.epochs

        # Create the learning rate scheduler.
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0,  # Default value in run_glue.py
                                                    num_training_steps=total_steps)

        # This training code is based on the `run_glue.py` script here:
        # https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

        # Set the seed value all over the place to make this reproducible.

        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        torch.cuda.manual_seed_all(self.config.seed)

        # Store the average loss after each epoch so we can plot them.
        loss_values = []

        # For each epoch...
        for epoch_i in range(0, self.config.epochs):

            self.train_epoch = epoch_i
            # ========================================
            #               Training
            # ========================================

            # Perform one full pass over the training set.

            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, self.config.epochs))
            print('Training...')

            # Measure how long the training epoch takes.
            t0 = time.time()

            # Reset the total loss for this epoch.
            total_loss = 0

            # Put the model into training mode. Don't be mislead--the call to
            # `train` just changes the *mode*, it doesn't *perform* the training.
            # `dropout` and `batchnorm` layers behave differently during training
            # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
            model.train()

            # For each batch of training data...
            for step, batch in enumerate(train_dataloader):

                # Progress update every 40 batches.
                if step % 40 == 0 and not step == 0:
                    # Calculate elapsed time in minutes.
                    elapsed = self.format_time(time.time() - t0)

                    # Report progress.
                    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

                # Unpack this training batch from our dataloader.
                #
                # As we unpack the batch, we'll also copy each tensor to the GPU using the
                # `to` method.
                #
                # `batch` contains three pytorch tensors:
                #   [0]: input ids
                #   [1]: attention masks
                #   [2]: labels
                b_input_ids = batch[0].to(self.config.device)
                b_input_mask = batch[1].to(self.config.device)
                b_labels = batch[2].to(self.config.device)

                # Always clear any previously calculated gradients before performing a
                # backward pass. PyTorch doesn't do this automatically because
                # accumulating the gradients is "convenient while training RNNs".
                # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
                model.zero_grad()

                # Perform a forward pass (evaluate the model on this training batch).
                # This will return the loss (rather than the model output) because we
                # have provided the `labels`.
                # The documentation for this `model` function is here:
                # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                outputs = model(b_input_ids,
                                token_type_ids=None,
                                attention_mask=b_input_mask,
                                labels=b_labels)

                # The call to `model` always returns a tuple, so we need to pull the
                # loss value out of the tuple.
                loss = outputs[0]

                # Accumulate the training loss over all of the batches so that we can
                # calculate the average loss at the end. `loss` is a Tensor containing a
                # single value; the `.item()` function just returns the Python value
                # from the tensor.
                total_loss += loss.item()

                # Perform a backward pass to calculate the gradients.
                loss.backward()

                # Clip the norm of the gradients to 1.0.
                # This is to help prevent the "exploding gradients" problem.
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                # Update parameters and take a step using the computed gradient.
                # The optimizer dictates the "update rule"--how the parameters are
                # modified based on their gradients, the learning rate, etc.
                optimizer.step()

                # Update the learning rate.
                scheduler.step()

            # Calculate the average loss over the training data.
            avg_train_loss = total_loss / len(train_dataloader)

            # Store the loss value for plotting the learning curve.
            loss_values.append(avg_train_loss)

            print("")
            print("  Average training loss: {0:.4f}".format(avg_train_loss))
            print("  Training epcoh took: {:}".format(self.format_time(time.time() - t0)))

            self.validate_on_dataset(model, validation_dataloader)

            if (self.config.model_out is not None) and ((epoch_i + 1) % 4 == 0):
                torch.save(model, self.config.model_out + "_" + str(epoch_i + 1) + ".pt")

        print("")
        print("Training complete!")

    def validate_on_dataset(self, model, validation_dataloader):
        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.

        print("")
        print("Running Validation...")

        t0 = time.time()

        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        model.eval()

        # Gold labels per batch
        labels_per_batch = []

        # Input sequence per batch
        input_ids_per_batch = []

        # Prediction weights per batch
        outputs_per_batch = []

        # Evaluate data for one epoch
        for batch in validation_dataloader:
            # Add batch to GPU
            batch = tuple(t.to(self.config.device) for t in batch)

            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch

            # Telling the model not to compute or store gradients, saving memory and
            # speeding up validation
            with torch.no_grad():
                # Forward pass, calculate logit predictions.
                # This will return the logits rather than the loss because we have
                # not provided labels.
                # token_type_ids is the same as the "segment ids", which
                # differentiates sentence 1 and 2 in 2-sentence tasks.
                # The documentation for this `model` function is here:
                # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                outputs = model(b_input_ids,
                                token_type_ids=None,
                                attention_mask=b_input_mask)

            labels_per_batch.append(b_labels)
            input_ids_per_batch.append(b_input_ids)
            outputs_per_batch.append(outputs)

        input_ids_list = self.convert_batch_of_input_ids_to_list(input_ids_per_batch)
        label_list = self.convert_batch_of_labels_to_list(labels_per_batch)
        logit_list = self.convert_batch_of_outputs_to_list_of_logits(outputs_per_batch)

        self.evaluate(input_ids_list, label_list, logit_list)
        self.write_results(input_ids_list, label_list, logit_list)

    def convert_batch_of_labels_to_list(self, labels_per_batch):
        label_list = []

        for b_labels in labels_per_batch:
            label_list += list(b_labels.to('cpu').numpy())

        return label_list

    def convert_batch_of_input_ids_to_list(self, input_ids_per_batch):
        input_id_list = []

        for b_input_ids in input_ids_per_batch:
            input_id_list += list(b_input_ids.to('cpu').numpy())

        return input_id_list

    def convert_batch_of_outputs_to_list_of_logits(self, output):
        logit_list = []

        for b_output in output:
            logit_list += list(b_output[0].detach().cpu().numpy())

        return logit_list

    def evaluate(self, input_id_list, label_list, logit_list):
        preds = np.argmax(logit_list, axis=1).flatten()
        self.evaluate_head(label_list, preds)

    def evaluate_head(self, y_true, y_pred):
        """
        Evaluate Precision, Recall, F1 scores between y_true and y_pred
        If output_file is provided, scores are saved in this file otherwise printed to std output.
        :param y_true: true labels
        :param y_pred: predicted labels
        :return: list of scores (F1, Recall, Precision, ExactMatch)
        """

        assert len(y_true) == len(y_pred)

        print()
        print("Accuracy: " + str(accuracy_score(y_true, y_pred)))
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1],
                                                                           average='weighted')
        scores = [
            "F1: %f\n" % f1,
            "Recall: %f\n" % recall,
            "Precision: %f\n" % precision,
            "ExactMatch: %f\n" % -1.0
        ]
        for s in scores:
            print(s, end='')

    def load_data(self, file, sampler=SequentialSampler):
        data = CSVTensorDataset(file, max_len=self.config.max_len, tokenizer_name=self.config.model_name)
        sampler = sampler(data)
        data_loader = DataLoader(data, sampler=sampler, batch_size=self.config.batch_size, num_workers=0)

        if self.num_labels == 0:
            self.num_labels = data.num_labels

        return data_loader

    def write_results(self, input_id_list, label_list, logit_list):
        self.write_results_on_head(input_id_list, label_list, logit_list)

    def write_results_on_head(self, input_id_list, label_list, logit_list, file_name_params=""):
        if self.config.out_file is not None:
            preds = np.argmax(logit_list, axis=1).flatten()

            pd.DataFrame(preds, columns=["pred"]).to_csv(self.config.out_file + str(self.train_epoch) + "." + file_name_params + ".csv", index=False)



