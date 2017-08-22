#################################
### Author: Paul Soto 		  ###
### 		paul.soto@upf.edu ###
#								#
# This file shows how to use a ##
# LSTM network with word vectors#
# to predict the next word in a #
# sequence. It is based off of ##
# the code from Nicolas Jimenez #
# https://github.com/nicodjimenez
#################################

import numpy as np 
import itertools

def loss(pred, label, deriv=False):
	if deriv:
		diff = (pred - label)
		return diff
	return 0.5*sum((pred - label) ** 2)

def sigmoid(x, deriv=False):
	if deriv:
		return x*(1-x) 
	return 1. / (1 + np.exp(-x))

def dtanh(x): 
	return 1. - x ** 2

def softmax(x, deriv=False):
	if deriv:
		vec = x.reshape((-1,1))
		jac = np.diag(x) - np.dot(vec, vec.T)
		return jac
	x = x-x.max()
	return np.exp(x)/np.exp(x).sum()

class LSTM:
	"""Class initiates an LSTM model with a forget, input and output gate
	and softmax activation function
	"""
	def __init__(self, x_size, c_size):
		"""
		x_size is size of input
		c_size is size of cell states and hidden states (which is output)
		"""
		self.x_size = x_size
		self.c_size = c_size
		# Theta matrices and bias terms for gates and output
		theta_g = np.random.rand(c_size,x_size+c_size)
		theta_i = np.random.rand(c_size,x_size+c_size)
		theta_f = np.random.rand(c_size,x_size+c_size)
		theta_o = np.random.rand(c_size,x_size+c_size)
		theta_h = np.random.rand(c_size,c_size)
		b_i = np.random.rand(c_size)
		b_f = np.random.rand(c_size)
		b_o = np.random.rand(c_size)
		b_g = np.random.rand(c_size)
		b_h = np.random.rand(c_size)
		self.params = {'theta_g':theta_g,'theta_i':theta_i, 'theta_f':theta_f, 
						'theta_o':theta_o,'theta_h':theta_h, 'b_i':b_i, 'b_f':b_f,
						'b_o':b_o, 'b_g':b_g, 'b_h':b_h}
		### Differentials for each parameter 
		self.deltas = {}
		for param in self.params.keys():
			self.deltas[param+"_delta"] = np.zeros_like(self.params[param])	

	def differentiate(self, lr = 1):
		### Apply the deltas to each weight 
		for param in self.params:
			self.params[param] -= lr*self.deltas[param+"_delta"]
			self.deltas[param+"_delta"] = np.zeros_like(self.params[param])

class LSTM_node:
	"""
	Class for forward propogated and backward propogating cell
	"""
	def __init__(self,param):
		self.param = param

	def forward_prop(self,x,h_prev,c_prev):
		"""
		x is input at time step t
		h_prev is hidden state (output) at time t-1
		c_prev is cell state at time t-1
		"""
		xh = np.hstack((x,h_prev))
		self.xh = xh
		self.i = sigmoid(np.dot(self.param.params["theta_i"],xh)+self.param.params["b_i"])
		self.f = sigmoid(np.dot(self.param.params["theta_f"],xh)+self.param.params["b_f"])
		self.o = sigmoid(np.dot(self.param.params["theta_o"],xh)+self.param.params["b_o"])
		self.g = np.tanh(np.dot(self.param.params["theta_g"],xh)+self.param.params["b_g"])
		self.c = self.f*c_prev+self.i*self.g
		self.z = self.o*np.tanh(self.c)
		self.h = softmax(np.dot(self.param.params['theta_h'],self.z)+self.param.params["b_h"])
		self.h_prev = h_prev
		self.c_prev = c_prev

	def back_prop(self,dh, dc_next):
		"""
		Performs gradient descent
		dh is the derivate of the loss function wrt to h 
		dc_next is the derivative of the loss function wrt to c_{t+1}
		"""
		dz = np.dot(softmax(self.h,deriv=True),dh)
		dc =np.dot(self.param.params['theta_h'].T,dz)*self.o*dtanh(self.c) + dc_next
		do = np.dot(self.param.params['theta_h'].T,dz)*np.tanh(self.c)*sigmoid(self.o,deriv=True)
		di = self.g*sigmoid(self.i,deriv=True)*dc
		df = self.c_prev*sigmoid(self.f, deriv=True)*dc
		dg = self.i*dtanh(self.g)*dc
		self.param.deltas['theta_o_delta'] += np.outer(do, self.xh)
		self.param.deltas['theta_g_delta']  += np.outer(dg, self.xh)
		self.param.deltas['theta_i_delta']  += np.outer(di, self.xh)
		self.param.deltas['theta_f_delta']  += np.outer(df, self.xh)
		self.param.deltas['theta_h_delta']  += np.outer(dz,self.o*np.tanh(self.c))
		self.param.deltas['b_i_delta']  += di
		self.param.deltas['b_f_delta'] += df      
		self.param.deltas['b_o_delta'] += do
		self.param.deltas['b_g_delta'] += dg  
		self.param.deltas['b_h_delta'] += dz  

		dxh = np.zeros_like(self.xh)
		dxh += np.dot(self.param.params['theta_i'].T, di)
		dxh += np.dot(self.param.params['theta_f'].T, df)
		dxh += np.dot(self.param.params['theta_o'].T, do)
		dxh += np.dot(self.param.params['theta_g'].T, dg)
		# Pass the dL/dc to time step at t-1
		# Pass the dL/dh to time step at t-1
		self.dc_prev = dc * self.f
		self.dh_prev = dxh[self.param.c_size:]

def predict_word(sentence, lstm_param):
	x_pred_list = []
	y_pred_list = []
	for sent in [sentence]:
		x_pred_list.append(map(lambda x: vectors[np.where(words==x)[0][0]],
				sent.split()))
	lstm_node_list = []
	# Initiate the LSTM cells and forward propogate
	for ex_ind in range(len(x_pred_list)):
		example = x_pred_list[ex_ind]
		for ind in range(len(example)):
			if len(lstm_node_list)<len(example):
				lstm_node_list.append(LSTM_node(param=lstm_param))
				if ind==0: 
					c_prev = np.zeros(c_size)
					h_prev = np.zeros(c_size)
				else:
					c_prev = lstm_node_list[ind-1].c
					h_prev = lstm_node_list[ind-1].h
				lstm_node_list[ind].forward_prop(example[ind],h_prev, c_prev)
	# Return largest probability word
	probs= lstm_node_list[-1].h
	probs_max = np.where(probs==max(probs))
	return predict[0],words[probs_max][0]


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
x_list = []
y_list = []
for sent in docs_split:
	if len(sent)<2:
		continue
	x_list.append(map(lambda x: vectors[np.where(words==x)[0][0]],sent[:-1]))
	y_list.append(map(lambda y: vectors[np.where(words==y)[0][0]],sent[1:]))

# Initiate the LSTM parameters
x_size  = vectors.shape[0]
c_size  = vectors.shape[0]
np.random.seed(0)
lstm_param = LSTM(x_size, c_size) 

# Training
lstm_node_list = []
losses = []
for epoch in range(500):
	print "epoch number %s" % epoch
	# Loop through each sentence
	for ex_ind in range(len(x_list)):
		example = x_list[ex_ind]
		output = y_list[ex_ind]
		# Each word in the sentence will be a timestep
		# Initiate the lstm_nodes
		for ind in range(len(example)):
			if len(lstm_node_list)<len(example):
				lstm_node_list.append(LSTM_node(param=lstm_param))
				if ind==0: 
					c_prev = np.zeros(c_size)
					h_prev = np.zeros(c_size)
				else:
					c_prev = lstm_node_list[ind-1].c
					h_prev = lstm_node_list[ind-1].h
				lstm_node_list[ind].forward_prop(example[ind],h_prev, c_prev)
		# Backpropogate and update the parameters in lstm_param
		idx = len(example)-1
		if ex_ind ==0: curr_loss = 0
		curr_loss += loss(lstm_node_list[idx].h,output[idx])
		dh_curr = loss(lstm_node_list[idx].h,output[idx],deriv=True)
		lstm_node_list[idx].back_prop(dh_curr, np.zeros(c_size))
		idx-=1
		while idx>=0:
			curr_loss += loss(lstm_node_list[idx].h,output[idx])
			dh = loss(lstm_node_list[idx].h,output[idx],deriv=True)
			dh += lstm_node_list[idx + 1].dh_prev
			dc = lstm_node_list[idx + 1].dc_prev
			lstm_node_list[idx].back_prop(dh, dc)
			idx-=1
		# To stop exploding gradient clip the delta values
		for param in lstm_param.deltas:
			lstm_param.deltas[param] = np.clip(lstm_param.deltas[param],-1,1)
		lstm_param.differentiate(lr=0.1)
		lstm_node_list = []
	losses.append(curr_loss)

# Let's see how well the model predicts the next word in the following sentences
print "'we': %s" % predict_word("we", lstm_param)[1]
print "'we think': %s" % predict_word("we think", lstm_param)[1]
print "'uncertainty and': %s" % predict_word("uncertainty and", lstm_param)[1]
print "'uncertainty and fears about': %s" % predict_word("uncertainty and fears about", lstm_param)[1]