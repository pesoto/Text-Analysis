#################################
### Author: Paul Soto 		  ###
### 		paul.soto@upf.edu ###
#								#
# This file is a class to #######
# run (uncollapsed) Gibbs       #
# sampling for Latent Dirichlet #
# Dirichlet Allocation  		#
#################################

import pandas as pd
from Text_Preprocessing import *
import re
from itertools import chain
import numpy as np 
from collections import Counter

class Gibbs():
	"""
	A class for the uncollapsed Gibbs sampling on text
	"""
	def __init__(self, text, K=2):
		"""
		text: Pandas series with each row a list of words in the document
		K: number of topics
		"""
		self.tokens = list(set(chain(*text.values)))
		self.V = len(self.tokens)
		self.K = K
		self.text = text

		### Create objects we'll need in updating parameters
		self.doc_topic = text.apply(lambda x: np.random.randint(0,self.K,size=(len(x),)))
		self.doc_topic_counts = self.doc_topic.apply(lambda x: Counter(x)).apply(pd.Series)
		self.doc_topic_counts = self.doc_topic_counts.fillna(0)

		# Fill missing columns (typically if K is too large)
		if list(self.doc_topic_counts.columns)!=range(self.K):
			needed = [el for el in range(self.K) if el not in self.doc_topic_counts.columns]
			for col in needed:
				self.doc_topic_counts[col] = 0

		self.term_topic_count = pd.DataFrame(index=self.tokens,columns=range(self.K),
											data=np.zeros((self.V,self.K)))
		for doc_ind in range(self.text.shape[0]):
			for (topic,word) in zip(self.doc_topic.ix[doc_ind],self.text.ix[doc_ind]):
				self.term_topic_count.loc[word,topic]+=1

		# Set priors
		self.alpha = 50.0/self.K
		self.beta = 200.0/self.V
		self.perplexity_scores = []

	def iterate(self,n=1000):
		"""
		Run n steps of the Gibbs sampler

		Relies on two calculations: 
			word_given_topic: "probability" of observing a word given a topic
			topic_given_doc: "probability" of observing topic j 

			Each is calculated by removing the current word from document
		"""
		for step in range(n):
			if step%25==0: 
				print "Step %s of Gibbs Sampling Completed" % step
				self.perplexity()
				print self.perplexity_scores

			for doc_ind,doc in enumerate(self.text):
				topics = self.doc_topic.ix[doc_ind]

				for word_ind,word in enumerate(doc):
					# Remove current word from current calculations
					self.doc_topic_counts.loc[doc_ind,topics[word_ind]]-=1
					self.term_topic_count.loc[word,topics[word_ind]]-=1

					# Find conditional probability 
					# Multiply how much a word likes a given topic by
					# 	how much a document likes that topic
					word_given_topic = (self.term_topic_count.ix[word]+self.beta)/\
										(self.doc_topic_counts.sum()+self.V*self.beta)
					topic_given_doc = (self.doc_topic_counts.ix[doc_ind]+self.alpha)/\
										(self.doc_topic_counts.sum(1).ix[doc_ind]+self.K*self.alpha)
					weights = word_given_topic*topic_given_doc
					weights = weights/weights.sum()

					new_topic = np.where(np.random.multinomial(1,weights)==1)[0][0]
					topics[word_ind] = new_topic

					# Add back the removed word to appropriate topic
					self.doc_topic_counts.loc[doc_ind,new_topic]+=1
					self.term_topic_count.loc[word,new_topic]+=1

				self.doc_topic.ix[doc_ind] = topics

	def perplexity(self):
		"""
		Compute perplexity scores of samples (currently insample)
		"""
		dt = (self.doc_topic_counts+self.alpha).apply(lambda x: x/x.sum(),1).fillna(0)
		tt = (self.term_topic_count+self.beta)/(self.term_topic_count+self.beta).sum().fillna(0)

		def prob(row):
			word_list = row[0]
			index = row['index']
			doc_perp= 0
			for each in word_list:
				doc_perp+=np.log((tt.ix[each]*dt.ix[index]).sum())
			return doc_perp

		perplexity = self.text.reset_index().apply(prob,1)
		perplexity = perplexity.sum()
		self.perplexity_scores.append(np.exp( - np.sum(perplexity) / self.text.apply(len).sum()))

	def top_n_words(self,n=10):
		"""
		Returns the n most frequent words from each topic
		"""

		for topic in range(self.K):
			top_n = self.term_topic_count.sort(topic,ascending=False)[topic].head(n)
			print "Top %s terms for topic %s" % (n,topic)

			for word in top_n.index: print word


data = [["rugby","football","competition","ball","games"],
        ["macro","economics","competition","games"],
        ["technology","computers","apple","AAPL","internet"],
        ["football","score","touchdown","team"],
        ["keynes","macro","friedman","policy"],
        ["stocks","AAPL","gains","analysis"],
        ["playoffs","games","season","compete","ball"],
        ["analysis","economy","economics","government"],
        ["apple","team","jobs","compete","computers"]]


gibbsobj = Gibbs(pd.Series(data),K=3)
gibbsobj.iterate()
