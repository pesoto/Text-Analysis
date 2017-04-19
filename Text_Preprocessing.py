#################################
### Author: Paul Soto 		  ###
### 		paul.soto@upf.edu ###
#								#
# This file is a class to #######
# perform the EM algorithm ######
# on a multinomial distribution##
#################################
import pandas as pd
from nltk import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk import download

# download('stopwords')

df = pd.read_csv("final_text_db.csv", encoding='utf-8')

class Preprocess():
	def __init__(self, text, sw=stopwords.words('english'), lower=True, stem = True):

		if not (type(text)==pd.core.series.Series):
			text = pd.Series(text)

		self.text = text
		self.sw = sw
		self.lower = lower
		self.stem = stem


	def clean_text(self):
		def stem(word_list):
			return map(lambda x: PorterStemmer().stem(x), word_list)

		def remove_sw(word_list):
			keep = []
			for word in word_list:
				if not word in self.sw:
					keep.append(word)
			return keep

		if self.lower:
			self.text = self.text.str.lower()

		self.text = self.text.apply(lambda x: x.split())
		
		if self.stem: self.text = self.text.apply(stem)
		if self.sw: self.text = self.text.apply(remove_sw)

		self.text = self.text.apply(lambda x: ' '.join(x))
		self.vectorizer = TfidfVectorizer()
		self.df_dense = self.vectorizer.fit_transform(self.text)

	def array(self, onehot=1):
		array = self.df_dense.toarray().copy()
		if onehot:
			array[array>0] = 1
		return array

	def make_df(self,onehot=1):
		df = pd.DataFrame(columns=self.vectorizer.get_feature_names(),
							data = self.array(onehot))
		df['Text'] = self.text
		df = df[['Text']+list(df.columns[:-1])]
		return df

docs = Preprocess(df.QUOTE)
docs.clean_text()
text_df = docs.make_df(onehot=1)