
我选这段代码的原因是：
1. 这是近期的作品
2. 这和 Machine Learning 跟 Natural Language Processing 相关
3. 这段代码的目的是对 StackOverflow 上的 QA 数据尽行内容分析

import pandas as pd                         #version 0.21.0
from tqdm import tqdm, tqdm_pandas
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim
import os
import datetime
import sys # for setting the recursion limit
import numpy as np
import re
# sys.setrecursionlimit(2000)

tokenizer = RegexpTokenizer(r'\w+')
# create English stop words list
# list_en_stop = get_stop_words('en')

list_en_stop = open("english.txt", "r").read().split('\n')

# convert to dictionary or it will take forever with O(n)
en_stop = {x:"" for x in list_en_stop}

# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()
p_stemmer2 = PorterStemmer()

# read the whole data and generate dataframes
# allEntries = pd.read_csv('./archive/allData.csv', encoding='latin-1') 
# allEntries = pd.read_csv('./archive/question2009.csv', encoding='latin-1') # for testing
# list for tokenized documents in loop

texts = []

if not os.path.isfile("parsedTexts2015.npy"):
    allEntries = pd.read_csv('./archive/question2015.csv', encoding='latin-1') # for testing
    dfallEntry = pd.DataFrame(allEntries, columns = ['Text', 'Score', 'AScore'])
    for counter, i in enumerate(tqdm(dfallEntry.Text)):
        raw = i.lower()
        tokens = tokenizer.tokenize(raw)
        stopped_tokens = [i for i in tokens if not i in en_stop]
        try:
            stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
        except:
            print('Recursion Error')
            stemmed_tokens = stopped_tokens    
        texts.append(stemmed_tokens)

    np.save("parsedTexts2015" ,texts)
else:
    texts = np.load("parsedTexts2015.npy").tolist()    

# turn our tokenized documents into a id <-> term dictionary
dictionary = corpora.Dictionary(texts)
    
# convert tokenized documents into a document-term matrix
corpus = [dictionary.doc2bow(text) for text in texts]

# generate LDA model
myNumTopic = 10 # number of topics to generate
myPass = 30 # number of passes
import time
start_time = time.time()

print(datetime.datetime.now())
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=myNumTopic, id2word = dictionary, passes=myPass)
# ldamodel = gensim.models.LdaMulticore(corpus, workers = 3, num_topics=myNumTopic, id2word = dictionary, passes=myPass)

print("--- %s seconds ---" % (time.time() - start_time))
directory = './LDAModel'
print('Outputing Model To...' + directory)

if not os.path.exists(directory):
    os.makedirs(directory)
ldamodel.save(directory + "/ldamodel_Pass" + str(myPass) + "_numTopic" + str(myNumTopic))    

################################################ update training ##########################################
ListOfDoc = []
for i in range(2008, 2017):
    ListOfDoc.append(i)
ListOfDoc.remove(2015)
for year in ListOfDoc:
    if not os.path.isfile("parsedTexts" + str(year) + ".npy"):
        filePath = './archive/question' + str(year) + '.csv'    
        allEntries = pd.read_csv(filePath, encoding='latin-1') # for testing
        dfallEntry = pd.DataFrame(allEntries, columns = ['Text', 'Score', 'AScore'])
        texts = []
        maxT = 0
        for counter, i in enumerate(tqdm(dfallEntry.Text)):
            # clean and tokenize document string
            raw = i.lower()
            tokens = tokenizer.tokenize(raw)
            # remove stop words from tokens
            stopped_tokens = [i for i in tokens if not i in en_stop]
            try:
                stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
            except:
                print('Recursion Error')
                stemmed_tokens = stopped_tokens
            texts.append(stemmed_tokens)    
        np.save("parsedTexts" + str(year) ,texts)
    else:
        texts = np.load("parsedTexts" + str(year) + ".npy").tolist()
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    # generate LDA model
    start_time = time.time()
    print(datetime.datetime.now())
    ldamodel.update(corpus, chunks_as_numpy=True)
    print("--- %s seconds ---" % (time.time() - start_time))
    directory = './LDAModel'
    print('Outputing Model To...' + directory)
    ldamodel.save(directory + "/ldamodel_Pass" + str(myPass) + "_numTopic" + str(myNumTopic))    

################################################ Printing Topics ##########################################


myText = []
for i in range(0, myNumTopic):
    myText.append(ldamodel.print_topic(i))
    print(str(i))


directory = './Results/'
print('Outputing Scores To...' + directory)
if not os.path.exists(directory):
    os.makedirs(directory)    

thefile = open(directory + 'LDATopicWords' + str(myNumTopic) + '_' + str(myPass) + '.txt', 'w')
for item in myText:
  thefile.write("%s\n" % item)


################################################ Calculate Score ##########################################

totalResult = np.zeros(len(ldamodel.get_topics()))
totalNumOfTopic = np.zeros(len(ldamodel.get_topics()))
for year in range(2008, 2017):
    filePath = './archive/question' + str(year) + '.csv'
    allEntries = pd.read_csv(filePath, encoding='latin-1') # for testing
    dfallEntry = pd.DataFrame(allEntries, columns = ['Text', 'Score', 'AScore'])    
    dfallEntry['WScore'] = dfallEntry.Score - dfallEntry.AScore
    texts = np.load("parsedTexts" + str(year) + ".npy").tolist()
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    dt = np.dtype('int,float')    
    weightedScore = np.zeros(len(ldamodel.get_topics()))
    weightedCount = np.zeros(len(ldamodel.get_topics()))
    evaluation = ldamodel[corpus]
    for i in tqdm(range(0, len(corpus))):
        array = np.array(evaluation[i], dtype = dt)        
        for counter, j in enumerate(array['f0']):
            weightedScore[j] += array['f1'][counter] * dfallEntry.WScore[i]
            weightedCount[j] += array['f1'][counter]
    totalResult = np.vstack((totalResult, weightedScore/weightedCount))
    totalNumOfTopic = np.vstack((totalNumOfTopic, weightedCount))
    # totalResult.append(weightedScore/weightedCount)


np.savetxt(directory + str(myNumTopic) + "_" + str(myPass) + "_YrVsTpScore.csv", totalResult, delimiter=",")
np.savetxt(directory + str(myNumTopic) + "_" + str(myPass) + "_YrVsTpCount.csv", totalNumOfTopic, delimiter=",")


################################################ Generate Keyword Report ##########################################

text_file = open('./Results/topKeyWords.txt', 'w')

for e in range(0, myNumTopic):
    text_file.write("----------------------------------- Topic %d -----------------------------------\n" % (e+1))
    a = ldamodel.print_topic(e, topn=10)
    matches = re.findall(r'\"(.+?)\"',a)
    output = '\n'.join(matches)
    text_file.write("%s\n\n\n" % output)

