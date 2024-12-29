CBOW Word2Vec Model with PyTorch

This project walks you through the process of implementing the Continuous Bag of Words (CBOW) model using Word2Vec and PyTorch. The primary objective is to train a neural network to generate word embeddings from a given dataset (Wikipedia), experiment with the embeddings, and test the performance on a word analogy dataset.

Project Overview:

We will follow these steps to implement the project:

1. Install Necessary Libraries:
We will need a few packages to load the dataset, preprocess text, and train the model.
We'll use PyTorch for building the model.

2. Dataset Preprocessing:
We'll load the Wikitext dataset (which is a subset of Wikipedia), preprocess it to clean and remove unnecessary elements, and generate the vocabulary and frequency distribution.

3. Vocabulary Creation:
Convert words into integer indices using a word-to-index dictionary.

4. Generate Training Data:
Using the context window and the word-to-index mapping, generate pairs of surrounding words (context) and target words for training.

5. Building the CBOW Model:
Implement the CBOW model which will take surrounding words and predict the center word (target word).

6. Training the Model:
Use Cross-Entropy Loss and the Adam optimizer to train the model.

7. Evaluating the Model:
Use word similarity metrics and the Google Analogy dataset to test the quality of the word embeddings.

8. Experimentation:
We experiment by changing the training corpus or testing on different datasets.
