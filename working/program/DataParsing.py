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
from nltk.tokenize import RegexpTokenizer
from gensim import corpora, models

import re # for parsing html tags
import os
# initialize progress display
tqdm_pandas(tqdm())


# Building Dictionary
stopWords = get_stop_words('en')
myDict = {}

myDict[r"\<(.*?)\>"] = ''

print('Loading Questions...')
questions = pd.read_csv('parsedQuestions.csv', encoding='latin-1') # parsedQuestions and parsedAnswers are small sample files
questions['Text'] = questions['Title'] + ' ' + questions['Body']
dfq = pd.DataFrame(questions, columns = ['Text', 'Score', 'CreationDate', 'Id'])


# define function
def chunker(seq, size):
    # from http://stackoverflow.com/a/434328
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))
def replace_with_progress(df, myDict):    
    chunksize = int(len(df) / 20) # 1%
    with tqdm(total=len(df)) as pbar:
        for cdf in chunker(df, chunksize):
            cdf['Text'].replace(myDict, regex = True, inplace = True)                        
            pbar.update(chunksize)            

# run replacement with progress
print('Deleting Question Stopwords...')
replace_with_progress(dfq, myDict)

# for word in stopWords: questions['Text'] = questions['Text'].str.replace(word, '')      #This takes a really long time, unsure why <-- stackoverflow says if you do the replace with dataframe you can cut the processtime in half

# parse into dataframe with text and score(upvote?) only
print('Loading Answers...')
answers = pd.read_csv('parsedAnswers.csv', encoding='latin-1')

# no need to parse the answer text. we never use it
dfa = pd.DataFrame(answers, columns = ['ParentId', 'Score'])

print('Merging Answers Votes to Questions...')
dfq['AScore'] = dfq['Id'].progress_apply(lambda x : dfa[dfa.ParentId == x].Score.sum())
dfq['AScore'].fillna(0, inplace=True) # replace Nan with 0


print('Outputing Cleaned Data...')
directory = './archive'
if not os.path.exists(directory):
    os.makedirs(directory)
for year in range(2008, 2017):
    newDF = dfq[dfq['CreationDate'].str.contains(str(year))]    
    filename = directory + '/question' + str(year).decode('utf-8') + '.csv'
    newDF.to_csv(filename, header=True, index=False, encoding='utf-8')
    print(str(year).encode('utf-8') + ' Saved')
filename = directory + '/allData.csv'
newDF.to_csv(filename, header=True, index=False, encoding='utf-8')


# Use LDA on Topics
# Documentation: http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html
# Another option for LDA is: https://radimrehurek.com/gensim/models/ldamodel.html






