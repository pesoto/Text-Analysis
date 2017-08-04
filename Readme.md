# Text-Analysis
Explaining textual analysis tools in Python. Including Preprocessing, Skip Gram (word2vec), and Topic Modelling. 

This is not a module for large scale use, but rather a set of scripts to explain the methodologies.

## Preprocessing ##

How are documents and words represented in Python? How can I clean text in Python by removing unnecessary words and adjusting for infrequent words?

Text_Preprocessing.py: explains common ways of representing text data in Python through one-hot encoded vectors, cleaning data with removal of stopwords and lowercasing, and TF-IDF weights.

## EM-Algorithm ##

How can I discover topics of documents? I.e. how can I calculate how much one article is about sports, another about business, etc.?

EM_Algorithm.py: explains how to estimate a distribution using the EM-Algorithm. This is a precursor to the topic modelling example.

## Gibbs Sampling ##

How can I discover topics of documents? I.e. how can I calculate how much one article is about sports, another about business, etc.?

Gibbs_Sampling.py: explains how Gibbs sampling works in the context of topic modelling. 

## Skip Gram ##

How can I find which words in my documents are related to each other syntactically and semantically? How does a basic neural network work?

Skip_Gram.py: explains how the Skip Gram model from Mikolov et al. works (with gradient descent and no negative sampling). 
