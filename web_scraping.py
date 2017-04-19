#################################
### Author: Paul Soto 		  ###
### 		paul.soto@upf.edu ###
#								#
# This file shows the basic of ##
# BeautifulSoup to datascrape ###
# an HTML based website and save#
# the data as a csv. 			#
#################################

from bs4 import BeautifulSoup
import urllib2
import pandas as pd

def get_HTML(url):
	"""
	This file creates a HTML soup from a given url
	"""
	req = urllib2.Request(url, headers={ 'User-Agent': 'Mozilla/5.0' })
	html = urllib2.urlopen(req).read()
	soup = BeautifulSoup(html, 'html.parser')
	return soup


# Get HTML Soup
afi_soup = get_HTML("http://www.afi.com/100Years/quotes.aspx")

# Isolate the table with the  
tables = afi_soup.find_all("table")
quote_table = tables[1]

# Loop through each row, retrieving the 
# quote ID number, quote, movie and year
first = True
for row in quote_table.find_all("tr"):
	# Get row elements
	row_elements = row.find_all('td')
	# Strip unnecessary text
	row_txt = map(lambda x: x.text.strip(),row_elements)
	if first:
		final_df = pd.DataFrame(columns=row_txt)
		first = False
		continue
	# Add to dataset
	final_df.loc[len(final_df)+1] = row_txt

# Clicking movie link to get the director
final_df['Director'] = ""
max_iter = 10
index = 0
for row in quote_table.find_all("tr"):
	if index>max_iter:
		break
	if row.find("a"):
		movie_soup = get_HTML(row.find("a")['href'])
		director = movie_soup.find(text="Director:").find_next().text
		final_df.loc[index,"Director"] = director
	index+=1

# Export to CSV
final_df.to_csv("movie_quotes.csv",index=False, encoding='utf-8')