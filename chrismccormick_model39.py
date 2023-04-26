import torch
import sys
import pandas as pd
import numpy as np
# If there's a GPU available...
if torch.cuda.is_available():

    # Tell PyTorch to use the GPU.
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


from transformers import BertTokenizer

# Load the BERT tokenizer.
print('Loading BERT tokenizer...')
tokenizer = BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased', do_lower_case=True)


def my_tokenize(comments, labels, max_len):
    '''
    Tokenize a dataset of comments.

    Parameters:
      `comments` - List of comments, represented as strings.
        `labels` - List of integer labels for the corresponding comments.
       `max_len` - Truncate all of the comments down to this length.

    Returns:
      `input_ids` - All of the comments represented as lists of token IDs,
                    padded out to `max_len`, and cast as a PyTorch tensor.
         `labels` - The labels for the corresponding comments, formatted as
                    a PyTorch tensor.
      `attention_masks` - PyTorch tensor with the same dimensions as
                          `input_ids`. For each token, simply indicates whether
                           it is padding or not.
    '''
    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []
    attention_masks = []

    print('Tokenizing {:,} comments...'.format(len(comments)))

    # For every comment ("sentence")...
    for sent in comments:

        # Report progress.
        if ((len(input_ids) % 500) == 0):
            print('  Tokenized {:,} comments.'.format(len(input_ids)))

        # `encode_plus` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the sentence to `max_length`
        #   (6) Create attention masks for [PAD] tokens.
        encoded_dict = tokenizer.encode_plus(
            sent,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=max_len,  # Pad & truncate all sentences.
            pad_to_max_length=True,
            truncation=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
        )

        # Add the encoded sentence to the list.
        input_ids.append(encoded_dict['input_ids'])

        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    print(input_ids)
    attention_masks = torch.cat(attention_masks, dim=0)

    # Convert the labels to a tensor.
    labels = torch.tensor(labels)

    return (input_ids, labels, attention_masks)


df_train = pd.read_csv('./data_results/mtsamples_cleaned_fix.csv', delimiter = '\t')#.query('Class == "14" or Class == "21" or Class == "26"  or Class == "32" or Class == "38" or Class == "23" or Class == "10"  or Class == "11" or Class == "16"  or Class == "22" ')
#df_train = df_train['Class']==2
#df_val = pd.read_csv('./data_results/results_val_binary.txt', delimiter = '\t')
df_train_data = df_train['sentence'].values
df_train_target = df_train['Class'].values


#df_val_data = df_val['sentence'].values
#df_val_target = df_val['Class'].values
max_len = 150

#(val_input_ids, val_labels, val_attention_masks) = my_tokenize(df_val_data, df_val_target, max_len = max_len)

#df_train_target=np.where(df_train_target==14, 0, df_train_target)
#df_train_target=np.where(df_train_target==21, 1, df_train_target)
#df_train_target=np.where(df_train_target==26, 2, df_train_target)
#df_train_target=np.where(df_train_target==32, 3, df_train_target)
#df_train_target=np.where(df_train_target==38, 4, df_train_target)
#df_train_target=np.where(df_train_target==23, 5, df_train_target)
#df_train_target=np.where(df_train_target==10, 6, df_train_target)
#df_train_target=np.where(df_train_target==11, 7, df_train_target)
#df_train_target=np.where(df_train_target==16, 8, df_train_target)
#df_train_target=np.where(df_train_target==22, 9, df_train_target)
#print(df_train_target)
(train_input_ids, train_labels, train_attention_masks) = my_tokenize(df_train_data, df_train_target, max_len = max_len)







import os
import csv


def check_gpu_mem():
    '''
    Uses Nvidia's SMI tool to check the current GPU memory usage.
    Reported values are in "MiB". 1 MiB = 2^20 bytes = 1,048,576 bytes.
    '''

    # Run the command line tool and get the results.
    buf = os.popen('nvidia-smi --query-gpu=memory.total,memory.used --format=csv')

    # Use csv module to read and parse the result.
    reader = csv.reader(buf, delimiter=',')

    # Use a pandas table just for nice formatting.
    df = pd.DataFrame(reader)

    # Use the first row as the column headers.
    new_header = df.iloc[0]  # grab the first row for the header
    df = df[1:]  # take the data less the header row
    df.columns = new_header  # set the header row as the df header

    # Display the formatted table.
    # display(df)

    return df


from transformers import BertForSequenceClassification, AdamW, BertConfig

# Load BertForSequenceClassification, the pretrained BERT model with a single
# linear classification layer on top.
model = BertForSequenceClassification.from_pretrained(
    "allenai/scibert_scivocab_uncased", # Use the 12-layer BERT model, with an uncased vocab.
    num_labels = 10, # For our 20 newsgroups!
    output_attentions = False, # Whether the model returns attentions weights.
    output_hidden_states = False, # Whether the model returns all hidden-states.
)

# Tell pytorch to run this model on the GPU.
desc = model.cuda()
check_gpu_mem()


from torch.utils.data import TensorDataset, random_split
dataset_train = TensorDataset(train_input_ids, train_attention_masks, train_labels)
#dataset_val = TensorDataset(val_input_ids, val_attention_masks, val_labels)
train_size = int(0.9 * len(dataset_train))
val_size = len(dataset_train) - train_size

# Divide the dataset by randomly selecting samples.
train_dataset, val_dataset = random_split(dataset_train, [train_size, val_size])
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
batch_size = 16

# Create the DataLoaders for our training and validation sets.
# We'll take training samples in random order.
train_dataloader = DataLoader(
            dataset_train,  # The training samples.
            #sampler = RandomSampler(dataset_train), # Select batches randomly
            batch_size = batch_size ,# Trains with this batch size.,
            shuffle=True
        )

# For validation the order doesn't matter, so we'll just read them sequentially.
validation_dataloader = DataLoader(
            val_dataset, # The validation samples.
            sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
            batch_size = batch_size*batch_size # Evaluate with this batch size.
        )

# Note: AdamW is a class from the huggingface library (as opposed to pytorch)
# I believe the 'W' stands for 'Weight Decay fix"
optimizer = AdamW(model.parameters(),
                  lr = 2e-5, # args.learning_rate - default is 5e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )

from transformers import get_linear_schedule_with_warmup

# Number of training epochs (authors recommend between 2 and 4)
epochs = 20

# Total number of training steps is number of batches * number of epochs.
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)

import numpy as np

labels_dist = {}
for i in range(0,10):
    labels_dist[i] = 0

pred_dist = {}
for i in range(0,10):
    pred_dist[i] = 0
# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    j = 0
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    for i in labels_flat:
        labels_dist[i] = labels_dist[i] +1
        if (i == pred_flat[j]):
            pred_dist[i] = pred_dist[i] +1
        j=j+1

    print(pred_dist)
    print(labels_dist)
    for i in labels_dist:
        print("Class ",i," got ", pred_dist[i]/labels_dist[i])
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


import time
import datetime


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


import random
import numpy as np

# This training code is based on the `run_glue.py` script here:
# https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

# Set the seed value all over the place to make this reproducible.
seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# We'll store a number of quantities such as training and validation loss,
# validation accuracy, and timings.
training_stats = []
import torch.nn as nn
softmax = nn.Softmax(dim=1)
# For each epoch...
for epoch_i in range(0, epochs):

    # ========================================
    #               Training
    # ========================================

    # Perform one full pass over the training set.

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    # Measure how long the training epoch takes.
    t0 = time.time()

    # Reset the total loss for this epoch.
    total_train_loss = 0

    # Put the model into training mode. Don't be mislead--the call to
    # `train` just changes the *mode*, it doesn't *perform* the training.
    # `dropout` and `batchnorm` layers behave differently during training
    # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
    model.train()

    # For each batch of training data...
    for step, batch in enumerate(train_dataloader):

        # Progress update every 50 batches.
        if step % 50 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)

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
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        # Check GPU memory for the first couple steps.
        if step < 2:
            print('\n  Step {:} GPU Memory Use:'.format(step))
            df = check_gpu_mem()
            print('    Before forward-pass: {:}'.format(df.iloc[0, 1]))

        # Always clear any previously calculated gradients before performing a
        # backward pass. PyTorch doesn't do this automatically because
        # accumulating the gradients is "convenient while training RNNs".
        # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
        model.zero_grad()

        # Perform a forward pass (evaluate the model on this training batch).
        # The documentation for this `model` function is here:
        # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
        # It returns different numbers of parameters depending on what arguments
        # arge given and what flags are set. For our useage here, it returns
        # the loss (because we provided labels) and the "logits"--the model
        # outputs prior to activation.
        loss, logits = model(b_input_ids,
                             token_type_ids=None,
                             attention_mask=b_input_mask,
                             labels=b_labels)

        # Report GPU memory use for the first couple steps.
        if step < 2:
            df = check_gpu_mem()
            print('     After forward-pass: {:}'.format(df.iloc[0, 1]))

        # Accumulate the training loss over all of the batches so that we can
        # calculate the average loss at the end. `loss` is a Tensor containing a
        # single value; the `.item()` function just returns the Python value
        # from the tensor.
        total_train_loss += loss.item()

        # Perform a backward pass to calculate the gradients.
        loss.backward()

        # Report GPU memory use for the first couple steps.
        if step < 2:
            df = check_gpu_mem()
            print('    After gradient calculation: {:}'.format(df.iloc[0, 1]))

        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update parameters and take a step using the computed gradient.
        # The optimizer dictates the "update rule"--how the parameters are
        # modified based on their gradients, the learning rate, etc.
        optimizer.step()

        # Update the learning rate.
        #scheduler.step()

    # Calculate the average loss over all of the batches.
    avg_train_loss = total_train_loss / len(train_dataloader)

    # Measure how long this epoch took.
    training_time = format_time(time.time() - t0)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(training_time))

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

    # Tracking variables
    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0

    # Evaluate data for one epoch
    for batch in validation_dataloader:
        # Unpack this training batch from our dataloader.
        #
        # As we unpack the batch, we'll also copy each tensor to the GPU using
        # the `to` method.
        #
        # `batch` contains three pytorch tensors:
        #   [0]: input ids
        #   [1]: attention masks
        #   [2]: labels
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        # Tell pytorch not to bother with constructing the compute graph during
        # the forward pass, since this is only needed for backprop (training).
        with torch.no_grad():
            # Forward pass, calculate logit predictions.
            # token_type_ids is the same as the "segment ids", which
            # differentiates sentence 1 and 2 in 2-sentence tasks.
            # The documentation for this `model` function is here:
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            # Get the "logits" output by the model. The "logits" are the output
            # values prior to applying an activation function like the softmax.
            (loss, logits) = model(b_input_ids,
                                   token_type_ids=None,
                                   attention_mask=b_input_mask,
                                   labels=b_labels)

        # Accumulate the validation loss.
        total_eval_loss += loss.item()

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Calculate the accuracy for this batch of test sentences, and
        # accumulate it over all batches.
        total_eval_accuracy += flat_accuracy(logits, label_ids)

    # Report the final accuracy for this validation run.
    avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
    print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

    # Calculate the average loss over all of the batches.
    avg_val_loss = total_eval_loss / len(validation_dataloader)

    validation_time = format_time(time.time() - t0)
    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    print("  Validation took: {:}".format(validation_time))

    # Record all statistics from this epoch.
    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Valid. Accur.': avg_val_accuracy,
            'Training Time': training_time,
            'Validation Time': validation_time
        }
    )
    print("Saving...")
    torch.save(model.state_dict(), './Models/clinical_{}_39_categories.dat'.format(epoch_i))

print("")
print("Training complete!")



sys.exit(0)