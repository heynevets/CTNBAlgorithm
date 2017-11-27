# COEN 281 Final Project
# CTNB Algorithm
# Authors:  Christian Ayscue
#           Ting-Yu Yeh
#           Nicholas Fong
#           Bing Tang
# Python version 3.6.3

# Import libraries
import numpy as np                          #version 1.13.3
import pandas as pd                         #version 0.21.0
from sklearn import decomposition           #version 0.19.1, used for LDA
from stop_words import get_stop_words       #version 2015.2.23.1, source: https://pypi.python.org/pypi/stop-words        Using this version instead of ntlk because ntlk doesn't have a 64 bit package in Windows
stopWords = get_stop_words('en')

# Read in data and do initial data cleanup
# Data source: https://www.kaggle.com/stackoverflow/stacksample/data
tags = pd.read_csv('Tags.csv', encoding='latin-1')
print(tags)

questions = pd.read_csv('Questions.csv', encoding='latin-1')
questions['Text'] = questions['Title'] + ' ' + questions['Body']
for word in stopWords: questions['Text'] = questions['Text'].str.replace(word, '')      #This takes a really long time, unsure why
# To-Do: Strip HTML Tags: https://stackoverflow.com/a/4869782/8839295      An alternative is to use the library BeautifulSoup (https://stackoverflow.com/a/12982689/8839295)
print(questions)

answers = pd.read_csv('Answers.csv', encoding='latin-1')
answers['Body'].apply(lambda x: [item for item in x if item not in stopWords])    #unsure if this version is faster than the above version for stripping out stopwords or not (https://stackoverflow.com/a/33246035/8839295)
print(answers)

# Use LDA on Topics
# Documentation: http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html
# Another option for LDA is: https://radimrehurek.com/gensim/models/ldamodel.html
