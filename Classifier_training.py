
import pandas as pd # provide sql-like data manipulation tools. very handy.
pd.options.mode.chained_assignment = None
import numpy as np # high dimensional vector computing library.
from copy import deepcopy
from string import punctuation
from random import shuffle
import json

import gensim
from gensim.models.word2vec import Word2Vec # the word2vec model gensim class
LabeledSentence = gensim.models.doc2vec.LabeledSentence # The input for a Doc2Vec model should be a list of TaggedDocument(['list','of','word'], [TAG_001])

from tqdm import tqdm
tqdm.pandas(desc="progress-bar") #seems to be for a good downloading interface

from nltk.tokenize import TweetTokenizer # a tweet tokenizer from nltk. EG : "This is a tweet :P #tweet " => "this" "is" "a" "tweet" ":P" "#tweet"
tokenizer = TweetTokenizer()

from sklearn.model_selection import train_test_split #split the data for training and validation
from sklearn.feature_extraction.text import TfidfVectorizer  #IDF = Inverse Document Frequency  
#is a weight indicating how commonly a word is used. 
# The more frequent its usage across documents, the lower its score. The lower the score, the less important the word becomes.
#TF-IDF term frequency-inverse document frequency , evaluate the importance of words in a dow
#the weight increases when the occurancy of a word increases as well IN the document

#With Tfidfvectorizer you compute the word counts, idf and tf-idf values all at once


#preprocessing part
"""
Below is a function that loads the dataset and extracts the two columns we need:
The sentiment: a binary (0/1) variable
The text of the tweet: string
"""

data = pd.read_csv('vire2.csv', encoding = 'UTF-16 LE',  sep = ';')
data = data[['Tweets_data', 'Manual_Sentiment_Annotation']]
#data = pd.read_csv('truc.csv', names= ['Sentiment', 'ItemID', 'SentimentSource', 'index' , 'name', 'SentimentText' ] )
#data = data[['Sentiment','SentimentText']]
print(data.head(5))

"""tokenizing function that splits each tweet into tokens and removes user mentions, hashtags and urls. 
These elements are very common in tweets but unfortunately they do not provide enough semantic information for the task"""

def tokenize(tweet):
    try:
        tweet = unicode(tweet.decode('utf-8').lower())
        tokens = tokenizer.tokenize(tweet)
        tokens = filter(lambda t: not t.startswith('@'), tokens)
        tokens = filter(lambda t: not t.startswith('#'), tokens)
        tokens = filter(lambda t: not t.startswith('http'), tokens)
        return tokens
    except:
        return 'NC'

#it's for tokenizing and cleaning all the data 
#data[tokens] is the tweet text but split into tokens
def postprocess(data, n=1000000):
    data = data.head(n)
    data['tokens'] = data['Tweets_data'].progress_map(tokenize)  ## progress_map is a variant of the map function plus a progress bar. Handy to monitor DataFrame creations.
    data = data[data.tokens != 'NC'] #remove lines with 'NC', resulting from a tokenization error (usually due to weird encoding
    # data.reset_index(inplace=True)
    # data.drop('index', inplace=True, axis=1)
    return data

# data = postprocess(data)
#print(data)
#Building the word2vec model
#divide the tokens (tweet text tokenised) and the Sentiment (0 or 1) for training and validation
n = 1000000 
#remettre .tokens si postprocess
x_train, x_test, y_train, y_test = train_test_split(np.array(data.head(n).Tweets_data),
                                                    np.array(data.head(n).Manual_Sentiment_Annotation), test_size=0.3)

#x_train ==> tweet text tokenized for training
#y_train ==> binary decision for training 


#turn tokens into LabeledSentence objects. Reminder :The input for a Doc2Vec model should be a list of TaggedDocument
def labelizeTweets(tweets, label_type): #tweets = x_train , x_test  
    labelized = []
    for i,v in tqdm(enumerate(tweets)): #as usual tqdm is for a good interface, 
        label = '%s_%s'%(label_type,i)  #will say : the first (1) tweet will be for TRAIN , the 2 is attached with TRAIN, the 3 with TRAIN....
        labelized.append(LabeledSentence(v, [label])) #labelized will be a list of tagged document for the model as input : the tweet tokenized for training/test with his lavel (0 , TRAIN)
    return labelized

x_train = labelizeTweets(x_train, 'TRAIN')
x_test = labelizeTweets(x_test, 'TEST')

x_train[0] #TaggedDocument(words=[u'thank', u'you', u'!', u'im', u'just', u'a', u'tad', u'sad', u'u', u'r', u'off', u'the', u'market', u'tho', u'...'], tags=['TRAIN_0']
#==> WE SEE HERE THAT we have a section WORDS in the list


n_dim = 200 
tweet_w2v = Word2Vec(size=n_dim, min_count=10) #the model is initialized with the dimension of the vector space 
#and min_count (a threshold for filtering words that appear less
tweet_w2v.build_vocab([x.words for x in tqdm(x_train)]) #we add all the section words of the tweet text tokenized (don't care of the tags) into the build_vocab
#build_vocab : uses all these tokens to internally create a vocabulary :a set of unique words .
# we are training a neural network where we train the model to predict the current word BASED ON THE CONTEXT 
tweet_w2v.train([x.words for x in tqdm(x_train)], epochs=tweet_w2v.iter, total_examples=tweet_w2v.corpus_count) #weights of the model are updated



#classifier
"""we have a word2vec model that converts each word from the corpus into a high dimensional vector. 
In order to classify tweets, we have to turn them into vectors as well. 
Since we know the vector representation of each word composing a tweet, 
we have to "combine" these vectors together and get a new one that represents the tweet as a whole.
A first approach consists in averaging the word vectors together. 
But a slightly better solution is to compute a weighted average where each weight gives the importance of the word with respect to the corpus. ==> tf-idf score
"""

print ('building tf-idf matrix ...')
vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=10) #
#When building the vocabulary ignore terms that have a document frequency strictly lower than the given threshold

matrix = vectorizer.fit_transform([x.words for x in x_train]) #the matrix is used where ?
tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
print ('vocab size :', len(tfidf))

#given a list of tweet tokens, creates an averaged tweet vector.
def buildWordVector(tokens, size): #size will be n_dim 
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in tokens:
        try:
            vec += tweet_w2v[word].reshape((1, size)) * tfidf[word] #on multiplie par son score
            count += 1.
        except KeyError: # handling the case where the token is not
                         # in the corpus. useful for testing.
            continue
    if count != 0:
        vec /= count #divide by all the tweets passed into like an average X)
    return vec


from sklearn.preprocessing import scale
train_vecs_w2v = np.concatenate([buildWordVector(z, n_dim) for z in tqdm(map(lambda x: x.words, x_train))]) 
train_vecs_w2v = scale(train_vecs_w2v) #scale each column to have zero mean and unit standard deviation.

test_vecs_w2v = np.concatenate([buildWordVector(z, n_dim) for z in tqdm(map(lambda x: x.words, x_test))])
test_vecs_w2v = scale(test_vecs_w2v) #scale each column to have zero mean and unit standard deviation.



from keras import Sequential
from keras.layers import Dense
from keras.models import model_from_json

model = Sequential()
model.add(Dense(32, activation='relu', input_dim=200))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(train_vecs_w2v, y_train, epochs=9, batch_size=32, verbose=2)



score = model.evaluate(test_vecs_w2v, y_test, batch_size=128, verbose=2)
print(score[1])

model_json = model.to_json()
with open("model_in_json.json", "w") as json_file:
    json.dump(model_json, json_file)

model.save_weights("model_weights.h5")
 