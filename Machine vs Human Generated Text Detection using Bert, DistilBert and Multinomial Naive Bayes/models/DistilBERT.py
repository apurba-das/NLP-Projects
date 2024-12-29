import json
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from transformers import DistilBertModel
from torch.utils.data import DataLoader
from tqdm import tqdm # shows progress bar
import numpy as np

"""
Download the necessary datasets from the following links:

# SemEval
- Training data: https://drive.google.com/file/d/1HeCgnLuDoUHhP-2OsTSSC3FXRLVoI6OG/view?usp=drive_link
- Development data: https://drive.google.com/file/d/1e_G-9a66AryHxBOwGWhriePYCCa4_29e/view?usp=drive_link
- Test data: https://drive.google.com/file/d/1-TN7sfSK1BuYHXlqxHHfwjEIE0JfarPk/view?usp=drive_link

# GenAI
- Training data: https://drive.google.com/file/d/1o8LE5p5xRdEFGrZOKiY4In2xW2BiWJbG/view?usp=drive_link
- Development data: https://drive.google.com/file/d/1hYIHqU3IMnJjPMTvl99K8pQUIOe7a957/view?usp=drive_link
"""

#--LOAD DATA--
# Get text data and corresponding labels from JSONL file (for train and validation sets)
def get_texts_labels(file_path):
    """
    Method to extract texts and labels from a JSONL file.
    Args:
    file_path : str : path to the JSONL file

    Returns:
    texts : list : list of text samples
    labels : list : list of labels corresponding to texts
    """
    texts = []
    labels = []

    # Open and read JSONL file
    with open(file_path, 'r') as f:
        for line in f:
            record = json.loads(line.strip())  # parse each line from a JSON-formatted string into a dictionary after removing leading and trailing whitespace or newline characters
            texts.append(record['text'])  # append 'text' field to texts list
            labels.append(record['label'])  # append 'label' field to labels list

    return texts, labels

# Path to JSONL files
# Please update the file paths according to where you saved the files on your computer
SemEval_train_file_path = 'subtaskA_train_monolingual.jsonl'  # SemEval train
SemEval_val_file_path = 'subtaskA_dev_monolingual.jsonl'      # SemEval val
SemEval_test_file_path = 'subtaskA_monolingual.jsonl'         # SemEval test
GenAI_train_file_path = 'en_train.jsonl'                         # GenAI train
GenAI_val_file_path = 'en_dev.jsonl'                             # GenAI val

# Create the datasets and corresponding labels
SemEval_train_texts, SemEval_train_labels = get_texts_labels(SemEval_train_file_path) # SemEval train
SemEval_val_texts, SemEval_val_labels = get_texts_labels(SemEval_val_file_path) # SemEval val
SemEval_test_texts, SemEval_test_labels = get_texts_labels(SemEval_test_file_path)  # SemEval test
GenAI_train_texts, GenAI_train_labels = get_texts_labels(GenAI_train_file_path) # GenAI train
GenAI_val_texts, GenAI_val_labels = get_texts_labels(GenAI_val_file_path) # GenAI val
# Randomly sample (via stratified sampling) GenAI train and val data so that it is a manageable subset for hyperparameter tuning (faster runtimes for quicker results)
# Split GenAI train data into new train and test sets
GenAI_train_data, GenAI_test_data = train_test_split(
    list(zip(GenAI_train_texts, GenAI_train_labels)),  # combine data and labels for sampling
    test_size=0.1,  # 10% for test
    stratify=GenAI_train_labels,  # maintain class balance
    random_state=42  # for reproducibility
)
# Unzip the new train and test sets
GenAI_train_texts, GenAI_train_labels = map(list, zip(*GenAI_train_data))
GenAI_test_texts, GenAI_test_labels = map(list, zip(*GenAI_test_data))

# Merge SemEval and GenAI train data
merged_train_texts = SemEval_train_texts + GenAI_train_texts
merged_train_labels = SemEval_train_labels + GenAI_train_labels

# Randomly sample (via stratified sampling) GenAI train and val data so that it is a manageable subset for hyperparameter tuning (faster runtimes for quicker results)
# Stratified sampling for GenAI train data
sampled_train_data, _ = train_test_split( # we are only interested in the train data portion, we don't need the other portion (which is usually test)
    list(zip(GenAI_train_texts, GenAI_train_labels)),  # combine data and labels for sampling
    train_size=len(SemEval_train_texts), # sample size based on SemEval train length
    stratify=GenAI_train_labels,  # stratify to maintain class balance (i.e. not have one class over-represented)
    random_state=42  # for reproducibility
)
# Unzip the sampled data and labels
GenAI_sampled_train_texts, GenAI_sampled_train_labels = map(list, zip(*sampled_train_data)) # note unzipped version returns a tuple, must convert it into a list to match rest of data manipulation

# Stratified sampling for GenAI val data
sampled_val_data, _ = train_test_split( # we are only interested in the train data portion (in this case, val), we don't need the other portion (which is usually test)
    list(zip(GenAI_val_texts, GenAI_val_labels)),  # combine data and labels for sampling
    train_size=len(SemEval_val_texts), # sample size based on SemEval val length
    stratify=GenAI_val_labels,  # stratify to maintain class balance (i.e. not have one class over-represented)
    random_state=42  # for reproducibility
)
# Unzip the sampled data and labels
GenAI_sampled_val_texts, GenAI_sampled_val_labels = map(list, zip(*sampled_val_data)) # note unzipped version returns a tuple, must convert it into a list to match rest of data manipulation


#--PREPROCESS DATA--
# Define BERT's tokenizer
"""
BERT's tokenizer converts each input text as follows:
    input_ids - indices corresponding to each token in sentence
    token_type_ids - identifies which sequence a token belongs to when there are more than one sequence in input text (note each sequence is treated independently)
    attention_mask - indicates whether a token should be attended (0) or not (1) (i.e. is it a meaningful token or just padding)
                     this is because BERT expects input sequences of same length, so shorter sentences are padded to match longest sequence
                     padding tokens are meaningless and can be ignored, which is what attention_masks allows the model to do
"""
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# Preprocess the texts and labels using BERT's tokenizer
def preprocess_function(texts, tokenizer, max_length=128):
    """
    Method to preprocess the data by tokenizing texts and preparing inputs.
    Args:
    texts : list : list of text samples
    labels : list : list of integer labels
    max_length : int : maximum sequence length for tokenized input, default=128
    Returns:
    inputs : dict : dictionary containing tokenized inputs
    """
    # Tokenize input texts
    inputs = tokenizer(
        texts, # texts to be tokenized
        padding=True, # add padding to shorter sentences for equal-length sequences
        max_length=max_length, # maximum length for each sequence
        truncation=True, # truncate sequences longer than max_length for equal-length sequences
        return_tensors='pt'  # return PyTorch tensors for model input to use PyTorch neural network frameworks
    )
    return inputs

# Tokenize and preprocess the training, validation, and test data
SemEval_train_data = preprocess_function(SemEval_train_texts, bert_tokenizer)
SemEval_val_data = preprocess_function(SemEval_val_texts, bert_tokenizer)
SemEval_test_data = preprocess_function(SemEval_test_texts, bert_tokenizer)
GenAI_train_data = preprocess_function(GenAI_train_texts, bert_tokenizer)
GenAI_val_data = preprocess_function(GenAI_val_texts, bert_tokenizer)
GenAI_test_data = preprocess_function(GenAI_test_texts, bert_tokenizer)
merged_train_data = preprocess_function(merged_train_texts, bert_tokenizer)
GenAI_sampled_train_data = preprocess_function(GenAI_sampled_train_texts, bert_tokenizer)
GenAI_sampled_val_data = preprocess_function(GenAI_sampled_val_texts, bert_tokenizer)

# Convert training, validation and test labels to PyTorch tensors for model input
SemEval_train_label_data = torch.tensor(SemEval_train_labels)
SemEval_val_label_data = torch.tensor(SemEval_val_labels)
SemEval_test_label_data = torch.tensor(SemEval_test_labels)
GenAI_train_label_data = torch.tensor(GenAI_train_labels)
GenAI_val_label_data = torch.tensor(GenAI_val_labels)
GenAI_test_label_data = torch.tensor(GenAI_test_labels)
merged_train_label_data = torch.tensor(merged_train_labels)
GenAI_sampled_train_label_data = torch.tensor(GenAI_sampled_train_labels)
GenAI_sampled_val_label_data = torch.tensor(GenAI_sampled_val_labels)

# --DEFINE MODEL--
class DistilBERT_Text_Classifier(nn.Module):
    """ Class to define the DistilBERT Text Classifier model
    Attributes
    ---------
    distilbert : DistilBertModel
      The pre-trained DistilBERT model
    num_classes : int
      Number of classes; In our case, it is 2 (Machine vs. Human)
    """
    def __init__(self, num_classes):
        super().__init__()
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')  # load pre-trained DistilBERT model
        self.hidden_dim = self.distilbert.config.hidden_size  # dimension of hidden layer (DistilBERT)
        self.linear = nn.Linear(self.hidden_dim, num_classes)  # linear layer for classification

    def forward(self, input_ids, attention_mask):
        # pass input through DistilBERT
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask) # note no token_type_ids unlike BERT. This is because DistilBERT pre-trained removes the NSP objective as NSP was found to contribute less to downstream performance
        cls_output = outputs.last_hidden_state[:, 0, :]
        # Extract [CLS] token (first token of last hidden layer) for classification
        # In DistilBERT, since it was not pre-trained on NSP objective, the [CLS] token represents
        # aggregated contextual information derived from self-attention mechanism during pre-training on the MLM objective.
        out = self.linear(cls_output)  # pass through linear layer
        return out  # return logit outputs

# --TRAIN AND VALIDATE--
#creation of dataloader for training
train_data = GenAI_train_data # chosen train data (SemEval or GenAI)
train_label_data = GenAI_train_label_data # chosen (labels of) train data (SemEval or GenAI)

batch_size = 16 # batch size

# DataLoader is a utility that handles batching and shuffling of data during training.
# this allows training/validation on data [batch_size] (input data, label) pairs at a time (simultaneously) instead one pair at a time in one epoch
train_dataloader=DataLoader(list(zip(train_data['input_ids'], train_data['token_type_ids'], train_data['attention_mask'],train_label_data)),batch_size=batch_size,shuffle=True) # dataloader for train
SemEval_val_dataloader=DataLoader(list(zip(SemEval_val_data['input_ids'], SemEval_val_data['token_type_ids'], SemEval_val_data['attention_mask'],SemEval_val_label_data)),batch_size=batch_size,shuffle=True) # dataloader for SemEval validation
GenAI_val_dataloader=DataLoader(list(zip(GenAI_val_data['input_ids'], GenAI_val_data['token_type_ids'], GenAI_val_data['attention_mask'],GenAI_val_label_data)),batch_size=batch_size,shuffle=True) # dataloader for GenAI validation

# Evaluate model on dataset
def evaluate(model, dataloader, device):
    model.eval()  # set model to evaluation mode
    correct = 0  # keep track of the number of texts that are predicted correctly
    total = 0  # keep track of the number of texts that are actually processed successfully

    all_predictions =[] #  store all predictions
    all_labels = [] # store all gold labels (true value)
    with torch.no_grad(): # no update to gradient (freezes weights), because we don't train model on val data (only want to extract predictions for testing)
        for input_ids, token_type_ids, attention_mask, label in tqdm(dataloader):
            input_ids, token_type_ids, attention_mask, label = input_ids.to(device), token_type_ids.to(device), attention_mask.to(device), label.to(device)  # moves data to device

            log_probs = model(input_ids, attention_mask) # feedforward: run forward function of model on input texts and outputs 2 logits (for each class) per text/sample in batch
            _, predictions = torch.max(log_probs, dim=1) # get predictions - finds the index of the maximum logit value which is the index of our predicted class (0 or 1)
            correct += (predictions == label).sum().item() # count how many correct predictions have been made for current batch
            total += len(label) # count how many total predictions have been made, which is basically the length of label tensor across batches

            # Store predictions and labels for metrics (we make sure to move to CPU first to convert tensor to numpy array, making a numpy array for each batch)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

    # Convert lists to numpy arrays for ease of operations
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    # Note: Human text: Label 0; Machine text: Label 1
    # Predicted class: Positive (1), Target class: Positive (0) => True positive
    true_positives = np.sum((all_predictions == 1) & (all_labels == 1))
    # Predicted class: Positive (1), Target class: Negative (0) => False positive
    false_positives = np.sum((all_predictions == 1) & (all_labels == 0))
    # Predicted class: Negative (0), Target class: Positive (1) => False negative
    true_negatives = np.sum((all_predictions == 0) & (all_labels == 0))
    # Predicted class: Negative (0), Target class: Negative (0) => True negative
    false_negatives = np.sum((all_predictions == 0) & (all_labels == 1))

    accuracy = correct / total if total != 0 else 0 # calculate accuracy
    precision = true_positives / (true_positives + false_positives) # calculate precision
    recall = true_positives / (true_positives + false_negatives) # calculate recall
    f1_score = 2 * precision * recall / (precision + recall) # calculate f1-score

    return accuracy, precision, recall, f1_score

# Initialize model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available()
                      else "cpu")
model = DistilBERT_Text_Classifier(2).to(device) # moves initialized model (binary classifier where Machine text = label 1 and Human text = label 0) to device
loss_function = nn.CrossEntropyLoss() # initialize loss function (which is Cross Entropy)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5) # better version of SGD (Stochastic Gradient Descent)
epochs = 3 # define maximum number of epochs

# Training loop
for epoch in range(epochs): # for each epoch
    model.train() # set model to train mode
    total_loss = 0 # initialize total loss

    for input_ids, token_type_ids, attention_mask, label in tqdm(train_dataloader): # for each batch of (input_ids, token_type_ids, attention_mask, label) tuples
        #TODO: create code for training our model

        # TRAIN
        input_ids, token_type_ids, attention_mask, label = input_ids.to(device), token_type_ids.to(device), attention_mask.to(device), label.to(device) # moves data to device
        optimizer.zero_grad() # resets optimizer (gradients) to ensure weights from previous batch are not added to the gradient calculated for current batch

        log_probs = model(input_ids, attention_mask) # feedforward: run forward function of model on input texts and outputs and outputs 2 logits (for each class) per text/sample in batch

        loss= loss_function(log_probs,label) # criterion - calculate loss between prediction and target, note CrossEntropyLoss function expects target values to be indices and first converts logits of outputs to probs using softmax to calculate: -Σylog(p)
        # loss function only considers predicted probability for the class (discards the rest)
        total_loss += loss.item() # add loss to total loss

        loss.backward() # backpropagate: compute gradients of loss with respect to weights
        optimizer.step() # update weights using computed gradients (gradient descent step)

    print(f"Epoch {epoch+1} loss: {total_loss/len(train_dataloader)}")

    # VALIDATE
    # SemEval
    accuracy, precision, recall, f1_score = evaluate(model, SemEval_val_dataloader, device)  # evaluate (via accuracy, precision, recall, f1-score) after each epoch using GenAI validation set
    print(f"SemEval validation accuracy: {accuracy:.4f}")
    print(f"SemEval validation precision: {precision:.4f}")
    print(f"SemEval validation recall: {recall:.4f}")
    print(f"SemEval validation F1-score: {f1_score:.4f}")
    # GenAI
    accuracy, precision, recall, f1_score = evaluate(model, GenAI_val_dataloader, device)  # evaluate (via accuracy, precision, recall, f1-score) after each epoch using GenAI validation set
    print(f"GenAI validation accuracy: {accuracy:.4f}")
    print(f"GenAI validation precision: {precision:.4f}")
    print(f"GenAI validation recall: {recall:.4f}")
    print(f"GenAI validation F1-score: {f1_score:.4f}")

# --TEST--
# Evaluate performance on test set
#creation of dataloader for testing
SemEval_test_dataloader=DataLoader(list(zip(SemEval_test_data['input_ids'], SemEval_test_data['token_type_ids'], SemEval_test_data['attention_mask'],SemEval_test_label_data)),batch_size=batch_size,shuffle=True)
GenAI_test_dataloader=DataLoader(list(zip(GenAI_test_data['input_ids'], GenAI_test_data['token_type_ids'], GenAI_test_data['attention_mask'],GenAI_test_label_data)),batch_size=batch_size,shuffle=True)

# SemEval
accuracy, precision, recall, f1_score = evaluate(model, SemEval_test_dataloader,
                                                 device)  # evaluate (via accuracy, precision, recall, f1-score) after each epoch using GenAI test set
print(f"SemEval test accuracy: {accuracy:.4f}")
print(f"SemEval test precision: {precision:.4f}")
print(f"SemEval test recall: {recall:.4f}")
print(f"SemEval test F1-score: {f1_score:.4f}")

# GenAI
accuracy, precision, recall, f1_score = evaluate(model, GenAI_test_dataloader,
                                                 device)  # evaluate (via accuracy, precision, recall, f1-score) after each epoch using GenAI test set
print(f"GenAI test accuracy: {accuracy:.4f}")
print(f"GenAI test precision: {precision:.4f}")
print(f"GenAI test recall: {recall:.4f}")
print(f"GenAI test F1-score: {f1_score:.4f}")

# Evaluate performance on training set
#creation of dataloader for training set
SemEval_train_dataloader=DataLoader(list(zip(SemEval_train_data['input_ids'], SemEval_train_data['token_type_ids'], SemEval_train_data['attention_mask'],SemEval_train_label_data)),batch_size=batch_size,shuffle=True)
GenAI_train_dataloader=DataLoader(list(zip(GenAI_train_data['input_ids'], GenAI_train_data['token_type_ids'], GenAI_train_data['attention_mask'],GenAI_train_label_data)),batch_size=batch_size,shuffle=True)

# SemEval
accuracy, precision, recall, f1_score = evaluate(model, SemEval_train_dataloader,
                                                 device)  # evaluate (via accuracy, precision, recall, f1-score) after each epoch using GenAI validation set
print(f"SemEval train accuracy: {accuracy:.4f}")
print(f"SemEval train precision: {precision:.4f}")
print(f"SemEval train recall: {recall:.4f}")
print(f"SemEval train F1-score: {f1_score:.4f}")

# GenAI
accuracy, precision, recall, f1_score = evaluate(model, GenAI_train_dataloader,
                                                 device)  # evaluate (via accuracy, precision, recall, f1-score) after each epoch using GenAI validation set
print(f"GenAI train accuracy: {accuracy:.4f}")
print(f"GenAI train precision: {precision:.4f}")
print(f"GenAI train recall: {recall:.4f}")
print(f"GenAI train F1-score: {f1_score:.4f}")

# Evaluate performance on validation set
# SemEval
accuracy, precision, recall, f1_score = evaluate(model, SemEval_val_dataloader,
                                                 device)  # evaluate (via accuracy, precision, recall, f1-score) after each epoch using GenAI validation set
print(f"SemEval validation accuracy: {accuracy:.4f}")
print(f"SemEval validation precision: {precision:.4f}")
print(f"SemEval validation recall: {recall:.4f}")
print(f"SemEval validation F1-score: {f1_score:.4f}")

# GenAI
accuracy, precision, recall, f1_score = evaluate(model, GenAI_val_dataloader,
                                                 device)  # evaluate (via accuracy, precision, recall, f1-score) after each epoch using GenAI validation set
print(f"GenAI validation accuracy: {accuracy:.4f}")
print(f"GenAI validation precision: {precision:.4f}")
print(f"GenAI validation recall: {recall:.4f}")
print(f"GenAI validation F1-score: {f1_score:.4f}")
