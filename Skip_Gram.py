#################################
### Author: Paul Soto 		  ###
### 		paul.soto@upf.edu ###
#								#
# This file is a script to run ##
# a Skip Gram Model using a toy #
# sample of documents. While ####
# negative sampling should be ###
# used to train the neural ######
# network, I use gradient #######
# descent to focus on the #######
# architecture rather than the ##
# optimal estimation ############
#################################

import numpy as np 
from collections import Counter
import re
import itertools
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd 

# M is the number of words to look (on one side) of each word for the context
M = 2
# H is the dimension of the hidden layer
H = 3

def sigma(vector, deriv=False):
	"""
	This function returns a vector evaluated using the sigmoid function

	vector: numpy array of real values
	deriv: if True, evaluate first derivate of sigmoid 
	"""
	if deriv:
		return sigma(vector)*(1-sigma(vector))
	else:
		return np.exp(vector)/np.exp(vector).sum()

def get_context(word_list):
	"""
	This function returns the 2*M words in the context of each word in 
	the list

	word_list: List of words
	M: global variable of the window size
	"""
	samples = []
	for word_ind in range(len(word_list)):
		context_inds = range(max(0,word_ind-M),
							min(word_ind+M+1,len(word_list)))
		context_inds.remove(word_ind)
		context = [word_list[el] for el in context_inds]
		samples.append((word_list[word_ind],context))
	return samples

docs = ["we think uncertainty about unemployment",
		"uncertainty and fears about inflation",
		"we think fears about unemployment",
		"we think fears and uncertainty about inflation and unemployment",
		"constant negative press covfefe"]


# Split each document into a list of words
docs_split = map(lambda x: x.split(),docs)
docs_words = list(itertools.chain(*docs_split))

# Find unique words across all documents
words = np.unique(docs_words)

# Generate a one hot encoded vector for each unique word
vectors = np.eye(words.shape[0])

# Initiate randomly V and W matrices
V = np.random.randn(H,words.shape[0])
W = np.random.randn(words.shape[0],H)

# Create list of all training examples
training = list(itertools.chain(*map(get_context,docs_split)))

log_likelihood = np.array([])
epochs = 10000
learning_rate = 0.001
tolerance = 0.001
discount = float(learning_rate)/epochs

for epoch in range(epochs):
	likelihood = 0
	for example in training:
		# Forward propogate word
		input_index = np.where(words==example[0])[0][0]
		l_input = vectors[input_index]
		l_hidden = np.dot(V,l_input)
		l_output = np.dot(W,l_hidden)
		l_output_a = sigma(l_output)
		errors = np.zeros(words.shape[0])
		# Compute the error for each word in context window
		for context in example[1]:
			output_index = np.where(words==context)[0][0]
			l_target= vectors[output_index]
			errors += (l_output_a-l_target)
		# Update the weights of V and W matrices
		delta2 = errors*sigma(l_output,True)
		W -= learning_rate*np.outer(delta2,l_hidden)
		V -= learning_rate*np.outer(np.dot(W.T,delta2),l_input)
		likelihood+=sum(map(np.log,l_output_a))
	log_likelihood=np.append(log_likelihood,likelihood)
	learning_rate -= discount
	if epoch<2: continue
	if (abs(likelihood-log_likelihood[-2])<tolerance):
		break
Plot out word embeddings and log-likelihood function
# Plot out word embeddings and log-likelihood function
fig = plt.figure()
ax = fig.add_subplot(1,2,1,projection="3d")
ax.scatter(V[0],V[1],V[2], alpha=0.3)
for i,txt in enumerate(words):
	ax.text(V[0][i],V[1][i],V[2][i],txt, size=10)
ax = fig.add_subplot(1,2,2)
ax.plot(log_likelihood)
plt.show()