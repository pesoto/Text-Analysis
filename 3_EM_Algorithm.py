#################################
### Author: Paul Soto 		  ###
### 		paul.soto@upf.edu ###
#								#
# This file is a class to #######
# perform the EM algorithm ######
# on a multinomial distribution##
#################################

import numpy as np


class EM_algo():
	""""
	Algorithm for the multinomial mixture model
	"""
	def __init__(self, data, K=2,
					prior_rhos=None, 
					prior_betas=None):
		self.D = data.shape[0]
		self.V = data.shape[1]
		self.K = K
		self.data = data
		if prior_rhos==None:
			self.rhos = np.full(self.K, 1/float(self.K))
		else:
			self.rhos = prior_rhos
		if prior_betas==None:
			self.betas = np.random.dirichlet(self.V*[1], self.K)
		else:
			self.betas =prior_betas 
		self.ll_history = []

	def E_step(self):

		def complete_ll(dt_matrix):
			"""
			Computes the total likelihood using
			Pr(w_d) = sigma_k rho_k * Pi_v (beta_k,v)

			and ll = 
			"""
			### Use log-sum trick
			dt_tempZ = dt_matrix - np.max(dt_matrix,1)[:,np.newaxis]
			ll = np.log(np.exp(dt_tempZ).sum(1)).sum()+dt_temp.max(1).sum()
			return ll

		### Get likelihood for each document-topic
		dt_temp = np.zeros((self.D,self.K))
		for doc in range(self.D):
			for topic in range(self.K):
				dt_temp[doc,topic] = np.log(self.rhos[topic]) + \
									(self.data[doc]*np.log(self.betas[topic])).sum()
		### Compute complete log-likelihood
		self.ll = complete_ll(dt_temp)
		self.ll_history.append(self.ll)

		### normalize the Zs
		self.zs = np.exp(dt_temp)/np.exp(dt_temp).sum(1)[:,np.newaxis]


	def M_step(self):
		### Update rhos
		self.rhos = self.zs.sum(0)/self.D
		### Update betas
		betas_temp = np.dot(self.zs.T,self.data)
		self.betas=betas_temp/betas_temp.sum(1)[:,np.newaxis]

	def iterate(self,tolerance=0.001,max_iter = 500):
		count = 0
		while True:
			self.E_step()
			self.M_step()
			if len(self.ll_history)==1:
				continue
			if (self.ll-self.ll_history[-2])<tolerance:
				print("Arrived at target tolerance!")
				break
			if (count>max_iter):
				print("Exceeded max_iterations!")
				break
			count+=1
