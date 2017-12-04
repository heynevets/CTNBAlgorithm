





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
myNumTopic = 20
myPass = 5
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

thefile = open('LDATopicWords' + str(myNumTopic) + '_' + str(myPass) + '.txt', 'w')
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
np.savetxt(str(myNumTopic) + "_" + str(myPass) + "_YrVsTpScore.csv", totalResult, delimiter=",")
np.savetxt(str(myNumTopic) + "_" + str(myPass) + "_YrVsTpCount.csv", totalNumOfTopic, delimiter=",")


################################################ Calculate Score ##########################################

# ldamodel[corpus[0]] * dfallEntry.WScore[0]
# ldamodel[corpus[0]][0]

# dt = np.dtype('int,float')
# array = np.array(a, dtype = dt)





################################################ Calculate Score ##########################################



#### Training More Models


# # generate LDA model
# myNumTopic = 100
# myPass = 10
# ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=myNumTopic, id2word = dictionary, passes=myPass)
# directory = './LDAModel'
# print('Outputing Model To...' + directory)
# ldamodel.save(directory + "/ldamodel_Pass" + str(myPass) + "_numTopic" + str(myNumTopic))    

# myNumTopic = 100
# myPass = 5
# ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=myNumTopic, id2word = dictionary, passes=myPass)
# directory = './LDAModel'
# print('Outputing Model To...' + directory)
# ldamodel.save(directory + "/ldamodel_Pass" + str(myPass) + "_numTopic" + str(myNumTopic))    


# myNumTopic = 20
# myPass = 30
# ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=myNumTopic, id2word = dictionary, passes=myPass)
# directory = './LDAModel'
# print('Outputing Model To...' + directory)
# ldamodel.save(directory + "/ldamodel_Pass" + str(myPass) + "_numTopic" + str(myNumTopic))    


# myNumTopic = 20
# myPass = 20
# ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=myNumTopic, id2word = dictionary, passes=myPass)
# directory = './LDAModel'
# print('Outputing Model To...' + directory)
# ldamodel.save(directory + "/ldamodel_Pass" + str(myPass) + "_numTopic" + str(myNumTopic))    

# myNumTopic = 50
# myPass = 20
# ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=myNumTopic, id2word = dictionary, passes=myPass)
# directory = './LDAModel'
# print('Outputing Model To...' + directory)
# ldamodel.save(directory + "/ldamodel_Pass" + str(myPass) + "_numTopic" + str(myNumTopic))    



















##test
# import gensim
# ldamodel = gensim.models.ldamodel.LdaModel.load('./LDAModel_100Topics/ldamodel_Pass1_numTopic100')
# ldamodel = gensim.models.ldamodel.LdaModel.load('./LDAModel/ldamodel_Pass1_numTopic50')
# for i in range(0, 10):
#     ldamodel.print_topic(i)





# import gensim
# a = gensim.models.ldamodel.LdaModel.load('./LDAModel/ldamodel_Pass2_numTopic100')
# for i in range(0, 10):
#     a.print_topic(i)


# texts2 = []

# raw = i.lower()
# tokens = tokenizer.tokenize(raw)

# # remove stop words from tokens
# stopped_tokens = [i for i in tokens if not i in en_stop]

# # stem tokens
# stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
# texts2.append(stemmed_tokens)
