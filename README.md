
# Expectation Maximization (EM) Algorithm Applied to Text

This tutorial summarizes how the EM algorithm can be applied to texts. It is based on Stephen Hansen's lecture notes <a>https://sekhansen.github.io/pdf_files/lecture2.pdf</a> and text-mining module <a>https://github.com/sekhansen/text-mining-tutorial<a/>.

[**Just Give Me the Code**](#gimmethecode)

First, suppose we have 3 documents:


```python
data = [["rugby","football","competition","ball","games"],
        ["macro","economics","io","competition","games","econometrics"],
        ["business","economics","stocks","bonds","NYSE"]]
```

For example, document 1 is a <i>sports</i> document, document 2 is related to <i>economics</i>, and document 3 is related to <i>business</i>. 

**Goal**
1. Topic Distribution <br>
The goal is to find the distribution of K topics. That is, in our corpus what is the probability a random document is related to topic 1 or topic 2 or ... topic K? This is represented by $\rho$=$(\rho_1,\rho_2,...\rho_K)$ where $1\geq\rho_k\geq0$. This is just a vector of size K. 
<br>
<br>
2. Word Distrubition <br>
Given a topic $k$, what is the probability that the document contains any word in our vocabulary of size V. This is represented by $\mathbf\beta_k$=($\beta_{1,k},\beta_{2,k},...,\beta_{V,k})$ where $1\geq\beta_{v,k}\geq0$. We will have K such vectors, so $\mathbf\beta$ will be a matrix of size K by V.
<br>
<br>

We will assume that the probability of observing a document $w_d$ is: $Pr(w_d | \rho, \mathbf\beta)=\Sigma_{k=1}^V(\rho_k \Pi_{v=1}^V\beta_{v,k}^{x_{v,d}})$, where $x_{v,d}$ is the occurences of word v in document d.


So the probability of observing document 1 is just: $Pr(["rugby","football","competition","ball","games"] | \rho, \mathbf\beta)=\rho_{1}(\beta_{ball,1}^{1}*\beta_{bonds,1}^{0}*...)+\rho_{2}(\beta_{ball,2}^{1}*\beta_{bonds,2}^{0}*...)$


We can represent the data in a frequency matrix where each row is a document, and each column is a word the vocabulary. The values correspond to the frequencies a word shows up in a document


```python
import pandas as pd
pd.set_option('display.expand_frame_repr', False)

feature_counts = pd.DataFrame(data=
                [[ 1.,  0.,  0.,  1.,  0.,  1.,  1.,  0.,  1.,  0.],                            
                 [ 0.,  0.,  0.,  1.,  1.,  0.,  1.,  1.,  0.,  0.],
                 [ 0.,  1.,  1.,  0.,  1.,  0.,  0.,  0.,  0.,  1.]], 
                columns=[u'ball', u'bonds', u'business', u'competition', u'economics',                   
                         u'football', u'games', u'macro', u'rugby', u'stocks'])

print(feature_counts)
```

       ball  bonds  business  competition  economics  football  games  macro  rugby  stocks
    0     1      0         0            1          0         1      1      0      1       0
    1     0      0         0            1          1         0      1      1      0       0
    2     0      1         1            0          1         0      0      0      0       1
    

The EM algorithm starts by guessing the probability that a document comes from topic $k$. Let's define the number of topics, K, and our initial guess of the distribution of each topic.

We import numpy as this module gives us a lot of useful functions like linear algebra operations and sampling from different types of distributions. 

<b>k</b> is the number of (unknown or latent) topics.

<b>rhos</b> ($\rho$) is the initial "guess" of the distribution of topics. The uniform distribution is a good starting point. We'll guess the probability of a document belonging to topic 1 is 50% and topic 2 is 50%. 

<b>betas</b> ($\mathbf\beta$) we will also guess , the probability matrix where word $v$ appears in topic $k$


```python
import numpy as np

K = 2

D = feature_counts.shape[0]

V = feature_counts.shape[1]

rhos = np.full(k,1/float(k))

betas = np.random.dirichlet(V*[1], K)
```

### E-Step###

First let's see how much each document likes each topic. We do this by computing each documents likelihood, given our initial guess of $\rho$ and $\beta$. We will log these likelihoods then exponentiate them later.

Log-Likelihood of document $w_d$:
$log(Pr(w_d | \rho, \mathbf\beta))=\Sigma_{k=1}^K(log(\rho_k)+ \Sigma_{v=1}^Vlog(\beta_{v,k})*{x_{v,d}})$


```python
dt_temp = np.zeros((D,k))
for doc in range(D):
    for topic in range(k):
        dt_temp[doc,topic] = np.log(rhos[topic]) + (feature_counts.values[doc]*np.log(betas[topic])).sum()
```

We are in the log world so we need to un-log, or exponentiate, the likelihoods to retrieve $\Sigma_{k=1}(\rho_k \Pi_{v=1}^V\beta_{v,k}^{x_{v,d}})$


```python
print(np.exp(dt_temp))
```

    [[  1.47750913e-06   1.11585535e-05]
     [  4.05101144e-07   8.74903957e-06]
     [  1.74182527e-06   4.72287999e-07]]
    

Note the extremely small numbers. We will eventually need to calculate the complete likelihood over all documents, that is $l(X|\rho\beta) = \Sigma_{d=1}^{D} log(\Sigma_{k=1}^K\rho_k \Pi_{v=1}^V\beta_{v,k}^{x_{v,d}})=\Sigma_{d=1}^{D} log(\Sigma_{k=1}^Ke^{(log(\rho_k)+ \Sigma_{v=1}^Vlog(\beta_{v,k})*{x_{v,d}})})$. Taking the logarithms of such small numbers, as in the print above, will lead to <b>numerical underflow</b>. Fortunately, we can use the <b>log-sum trick</b>, which is factoring out the largest exponent:

$log\Sigma_{c}e^{b_c} = log[(\Sigma_{c}e^{b_c-B})e^B]=[log(\Sigma_ce^{b_c-B})]+B$, where $B$ is the largest $b_c$


```python
dt_tempZ = dt_temp - np.max(dt_temp,1)[:,np.newaxis]

print(dt_tempZ)

ll = np.log(np.exp(dt_tempZ).sum(1)).sum()+dt_temp.max(1).sum()

print(ll)
```

    [[-2.02184869  0.        ]
     [-3.07256244  0.        ]
     [ 0.         -1.30509988]]
    -35.9009185218
    

### M-Step###

Now we need to update our initial guesses of $\rho$ and $\beta$. 

1. First, let's update $\rho$. 

We will use the formula:

$\rho_k^{new}=\frac{\Sigma_d\hat{z_{d,k}}}{\Sigma_k\Sigma_d\hat{z_{d,k}}}$ where $\hat{z_{d,k}}=\frac{\rho_k \Pi_{v=1}^V\beta_{v,k}^{x_{v,d}}}{\Sigma_d \rho_k \Pi_{v=1}^V\beta_{v,k}^{x_{v,d}}}$


```python
probs = np.exp(dt_temp)/np.exp(dt_temp).sum(1)[:,np.newaxis]

print(probs)
```

    [[ 0.11692797  0.88307203]
     [ 0.04425332  0.95574668]
     [ 0.78669203  0.21330797]]
    

Each row shows the distribution of a document across the K topics, where the rows sum to 1. For example, the top left element is how much ["rugby","football","competition","ball","games"] likes topic 1. 

We can just average each column across rows to come up with the new values of $\rho$.


```python
rhos = probs.sum(0)/D

print(rhos)
```

    [ 0.31595777  0.68404223]
    

2. Second, let's update $\beta$

We will use the formula 
$\beta_{k,v}^{new}=\frac{\Sigma_d\hat{z_{d,k}}*x_{d,v}}{\Sigma_d\hat{z_{d,k}}*\Sigma_vx_{d,v}}$




```python
betas_temp = np.dot(probs.T,feature_counts.values)
betas=betas_temp/betas_temp.sum(1)[:,np.newaxis]

print(betas)
```

    [[ 0.02991693  0.20128128  0.20128128  0.04123949  0.21260384  0.02991693
       0.04123949  0.01132256  0.02991693  0.20128128]
     [ 0.09713077  0.02346215  0.02346215  0.20225516  0.12858654  0.09713077
       0.20225516  0.10512439  0.09713077  0.02346215]]
    

And that's one iteration of the algorithm. The E-step and the M-step would be repeated until the log-likehood increases very litle. We can implement the entire code in the form of a class as follows:

<a id='gimmethecode'></a>


```python
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
		### Get likelihood for each document-topic
		dt_temp = np.zeros((self.D,self.K))
		for doc in range(self.D):
			for topic in range(self.K):
				dt_temp[doc,topic] = np.log(self.rhos[topic]) + \
									(self.data[doc]*np.log(self.betas[topic])).sum()


		### Use log-sum trick
		dt_tempZ = dt_temp - np.max(dt_temp,1)[:,np.newaxis]

		### Compute complete log-likelihood
		ll = np.log(np.exp(dt_tempZ).sum(1)).sum()+dt_temp.max(1).sum()
		### normalize the Zs
		self.zs = np.exp(dt_temp)/np.exp(dt_temp).sum(1)[:,np.newaxis]
		self.ll = ll
		self.ll_history.append(ll)

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

```


```python
em_instance = EM_algo(feature_counts.values,K=2)

em_instance.iterate()

print(em_instance.rhos)
print(em_instance.betas)
```

    Arrived at target tolerance!
    [ 0.33333333  0.66666667]
    [[  0.00000000e+000   2.50000000e-001   2.50000000e-001   3.18096923e-141
        2.50000000e-001   0.00000000e+000   3.18096923e-141   3.18096923e-141
        0.00000000e+000   2.50000000e-001]
     [  1.11111111e-001   6.39822930e-091   6.39822930e-091   2.22222222e-001
        1.11111111e-001   1.11111111e-001   2.22222222e-001   1.11111111e-001
        1.11111111e-001   6.39822930e-091]]
    
