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
from tqdm import tqdm, tqdm_pandas

# initialize progress display
tqdm_pandas(tqdm())


# Building Dictionary
stopWords = get_stop_words('en')
myDict = {}
for word in stopWords:
    myDict[word] = ''
myDict['<p>'] = ''
myDict['</p>'] = ''


# Read in data and do initial data cleanup
# Data source: https://www.kaggle.com/stackoverflow/stacksample/data

# tags = pd.read_csv('Tags.csv', encoding='latin-1')
# print(tags)

questions = pd.read_csv('Questions.csv', encoding='latin-1')
questions['Text'] = questions['Title'] + ' ' + questions['Body']
dfq = pd.DataFrame(questions, columns = ['Text', 'Score', 'CreationDate', 'Id'])
dfq['Text'].replace(myDict, regex = True, inplace = True)
# for word in stopWords: questions['Text'] = questions['Text'].str.replace(word, '')      #This takes a really long time, unsure why <-- stackoverflow says if you do the replace with dataframe you can cut the processtime in half

# To-Do: Strip HTML Tags: https://stackoverflow.com/a/4869782/8839295      An alternative is to use the library BeautifulSoup (https://stackoverflow.com/a/12982689/8839295)

# parse into dataframe with text and score(upvote?) only
answers = pd.read_csv('Answers.csv', encoding='latin-1')

# no need to parse the answer text. we never use it
# answers['Body'].progress_apply(lambda x: [item for item in x if item not in stopWords])    #unsure if this version is faster than the above version for stripping out stopwords or not (https://stackoverflow.com/a/33246035/8839295) so so so slowwww

dfa = pd.DataFrame(answers, columns = ['ParentId', 'Score'])

#dfq['AScore']= map(lambda p, D: EOQ(D,p,ck,ch),df['p'], df['D']) 
dfq['AScore'] = dfq['Id'].progress_apply(lambda x : dfa[dfa.ParentId == x].Score.sum())
dfq['AScore'].fillna(0, inplace=True) # replace Nan with 0

#dfq['Id'].progress_apply(lambda x : dfa[dfa.ParentId == x].Score.sum())

for year in range(2008, 2017):
    newDF = dfq[dfq['CreationDate'].str.contains(str(year))]
    newDF.to_csv('question' + str(year) + '.csv')


# Use LDA on Topics
# Documentation: http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html
# Another option for LDA is: https://radimrehurek.com/gensim/models/ldamodel.html
