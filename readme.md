# Text-Analysis

This is not a module for large scale use, but rather a set of scripts to explain popular methodologies in text analysis, including Web Scraping, Preprocessing, Skip Gram (word2vec), and Topic Modelling. 

## 1. Web Scraping ##

How can I download text data from a website algorithmically using Python? How do I store the data in a csv file for later use?

Web_Scraping.py: explains how to download movie quotes and store the data neatly in a table using the Pandas Python module.

## 2. Preprocessing ##

How are documents and words represented in Python? How can I clean text in Python by removing unnecessary words and adjusting for infrequent words?

Text_Preprocessing.py: explains common ways of representing text data in Python through one-hot encoded vectors, cleaning data with removal of stopwords and lowercasing, and TF-IDF weights.

## 3. EM-Algorithm ##

How can I discover topics of documents? I.e. how can I calculate how much one article is about sports, another about business, etc.?

EM_Algorithm.py: explains how to estimate a distribution using the EM-Algorithm. This is a precursor to the topic modelling example.

## 4. Gibbs Sampling ##

How can I discover topics of documents? I.e. how can I calculate how much one article is about sports, another about business, etc.?

Gibbs_Sampling.py: explains how Gibbs sampling works in the context of topic modelling. 

## 5. Skip Gram ##

How can I find which words in my documents are related to each other syntactically and semantically? How does a basic neural network work?

Skip_Gram.py: explains how the Skip Gram model from Mikolov et al. works (with gradient descent and no negative sampling). 
