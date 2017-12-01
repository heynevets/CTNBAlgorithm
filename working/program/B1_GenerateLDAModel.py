





import pandas as pd                         #version 0.21.0
from tqdm import tqdm, tqdm_pandas
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim
import os

tokenizer = RegexpTokenizer(r'\w+')
# create English stop words list
# list_en_stop = get_stop_words('en')

list_en_stop = open("english.txt", "r").read().split('\n')

# convert to dictionary or it will take forever with O(n)
en_stop = {x:"" for x in list_en_stop}

# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()

# read the whole data and generate dataframes
# allEntries = pd.read_csv('./archive/allData.csv', encoding='latin-1') 
allEntries = pd.read_csv('./archive/question2008.csv', encoding='latin-1') # for testing
dfallEntry = pd.DataFrame(allEntries, columns = ['Text', 'Score', 'AScore'])





# create sample documents
# doc_a = "Brocolli is good to eat. My brother likes to eat good brocolli, but not my mother."
# doc_b = "My mother spends a lot of time driving my brother around to baseball practice."
# doc_c = "Some health experts suggest that driving may cause increased tension and blood pressure."
# doc_d = "I often feel pressure to perform well at school, but my mother never seems to drive my brother to do better."
# doc_e = "Health professionals say that brocolli is good for your health." 

# compile sample documents into a list
# doc_set = [doc_a, doc_b, doc_c, doc_d, doc_e]

# list for tokenized documents in loop


texts = []

# loop through document list
# for i in doc_set:
for i in tqdm(dfallEntry.Text):
    # clean and tokenize document string
    raw = i.lower()
    tokens = tokenizer.tokenize(raw)
    # remove stop words from tokens
    stopped_tokens = [i for i in tokens if not i in en_stop]
    # stem tokens
    stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
    # add tokens to list
    texts.append(stemmed_tokens)


# turn our tokenized documents into a id <-> term dictionary
dictionary = corpora.Dictionary(texts)
    
# convert tokenized documents into a document-term matrix
corpus = [dictionary.doc2bow(text) for text in texts]

# generate LDA model
myNumTopic = 100
myPass = 20
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=myNumTopic, id2word = dictionary, passes=myPass)

directory = './LDAModel'
print('Outputing Model To...' + directory)

if not os.path.exists(directory):
    os.makedirs(directory)
ldamodel.save(directory + "/ldamodel_Pass" + str(myPass) + "_numTopic" + str(myNumTopic))    

for i in range(0, myNumTopic):
    ldamodel.print_topic(i)
# test
import gensim
a = gensim.models.ldamodel.LdaModel.load('./LDAModel/ldamodel_Pass20_numTopic100')
for i in range(0, 100):
    a.print_topic(i)






# texts2 = []

# raw = i.lower()
# tokens = tokenizer.tokenize(raw)

# # remove stop words from tokens
# stopped_tokens = [i for i in tokens if not i in en_stop]

# # stem tokens
# stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
# texts2.append(stemmed_tokens)
