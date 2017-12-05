import pandas as pd                         #version 0.21.0
from tqdm import tqdm, tqdm_pandas
from gensim import corpora, models
import gensim
import os
import sys # for setting the recursion limit
import numpy as np

# ldamodel = gensim.models.ldamodel.LdaModel.load('./LDAModel/ldamodel_Pass1_numTopic50')
ldamodel = gensim.models.ldamodel.LdaModel.load('./LDAModel/ldamodel_Pass2_numTopic50')

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
    
np.savetxt("50_2_YrVsTpScore.csv", totalResult, delimiter=",")
np.savetxt("50_2_YrVsTpCount.csv", totalNumOfTopic, delimiter=",")