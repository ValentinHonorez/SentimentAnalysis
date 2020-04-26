
import pandas as pd # provide sql-like data manipulation tools. very handy.
pd.options.mode.chained_assignment = None
import numpy as np # high dimensional vector computing library.
from copy import deepcopy
from string import punctuation
from random import shuffle

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
def ingest():
    data = pd.read_csv('truc.csv')
    #data.drop(['ItemID', 'SentimentSource'], axis=1, inplace=True) #we don't need these
    data = data[data.Sentiment.isnull() == False] 
    #we take the values without NaN
    data['Sentiment'] = data['Sentiment'].map(int)
    data = data[data['SentimentText'].isnull() == False] 
    #we don't takes the NaN values
    data.reset_index(inplace=True) 
    #use the default index, inplace = True is for not copying the database
    data.drop('index', axis=1, inplace=True) 
    print 'dataset loaded with shape', data.shape    
    return data

#data = ingest()
data = pd.read_csv('truc.csv', names= ['Sentiment', 'ItemID', 'SentimentSource', 'index' , 'name', 'SentimentText' ] )
data = data[['Sentiment','SentimentText']]
data.head(5)

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
    data['tokens'] = data['SentimentText'].progress_map(tokenize)  ## progress_map is a variant of the map function plus a progress bar. Handy to monitor DataFrame creations.
    data = data[data.tokens != 'NC'] #remove lines with 'NC', resulting from a tokenization error (usually due to weird encoding
    data.reset_index(inplace=True)
    data.drop('index', inplace=True, axis=1)
    return data

data = postprocess(data)

#Building the word2vec model
#divide the tokens (tweet text tokenised) and the Sentiment (0 or 1) for training and validation
n = 1000000
x_train, x_test, y_train, y_test = train_test_split(np.array(data.head(n).tokens),
                                                    np.array(data.head(n).Sentiment), test_size=0.2)

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



#tweet_w2v['good'] #convert words to vectors (dimension = n_dim).
#tweet_w2v.most_similar('good') # Given a word, this method returns the top n similar ones, it's a probability to be closer to that given word in most of the tweets.
#Eg : 
#[(u'goood', 0.7355118989944458),
#  (u'great', 0.7164269685745239),
#  (u'rough', 0.656904935836792),
#  (u'gd', 0.6395257711410522),
#  (u'goooood', 0.6351571083068848),
#  (u'tough', 0.6336284875869751),
#  (u'fantastic', 0.6223267316818237),
#  (u'terrible', 0.6179217100143433),
#  (u'gooood', 0.6099461317062378),
#  (u'gud', 0.6096700429916382)]


#plot
#visualize these word vectors.
#first have to reduce their dimension to 2 using t-SNE. 
#Then, using an interactive visualization tool such as Bokeh, map them directly on 2D plane and interact with them.

import bokeh.plotting as bp
from bokeh.models import HoverTool, BoxSelectTool
from bokeh.plotting import figure, show, output_notebook

# defining the chart
# output_notebook()
# plot_tfidf = bp.figure(plot_width=700, plot_height=600, title="A map of 10000 word vectors",
#     tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave",
#     x_axis_type=None, y_axis_type=None, min_border=1)

# # getting a list of word vectors. limit to 10000. each is of 200 dimensions
# word_vectors = [tweet_w2v[w] for w in tweet_w2v.wv.vocab.keys()[:5000]] #tweets trained wv ==> keys like a dict and inside it, there is a dict : vocab and we take the keys

# # dimensionality reduction. converting the vectors to 2d vectors
# from sklearn.manifold import TSNE
# tsne_model = TSNE(n_components=2, verbose=1, random_state=0)
# tsne_w2v = tsne_model.fit_transform(word_vectors)
# #fit transform : To center the data (make it have zero mean and unit standard error), you subtract the mean and then divide the result by the standard deviation.
# #You do that on the training set of data. But then you have to apply the same transformation to your testing set (e.g. in cross-validation),
# #or to newly obtained examples before forecast. But you have to use the same two parameters mu and sigma (values) 
# # that you used for centering the training set.

# #putting everything in a dataframe
# tsne_df = pd.DataFrame(tsne_w2v, columns=['x', 'y'])
# tsne_df['words'] = tweet_w2v.wv.vocab.keys()[:5000]

# #plotting. the corresponding word appears when you hover on the data point.
# plot_tfidf.scatter(x='x', y='y', source=tsne_df)
# hover = plot_tfidf.select(dict(type=HoverTool))
# hover.tooltips={"word": "@words"}
# show(plot_tfidf)



#classifier
"""we have a word2vec model that converts each word from the corpus into a high dimensional vector. 
In order to classify tweets, we have to turn them into vectors as well. 
Since we know the vector representation of each word composing a tweet, 
we have to "combine" these vectors together and get a new one that represents the tweet as a whole.
A first approach consists in averaging the word vectors together. 
But a slightly better solution is to compute a weighted average where each weight gives the importance of the word with respect to the corpus. ==> tf-idf score
"""

print 'building tf-idf matrix ...'
vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=10) #
#When building the vocabulary ignore terms that have a document frequency strictly lower than the given threshold

matrix = vectorizer.fit_transform([x.words for x in x_train]) #the matrix is used where ?
tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
print 'vocab size :', len(tfidf)

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
train_vecs_w2v = np.concatenate([buildWordVector(z, n_dim) for z in tqdm(map(lambda x: x.words, x_train))]) #ask kevin
train_vecs_w2v = scale(train_vecs_w2v) #scale each column to have zero mean and unit standard deviation.

test_vecs_w2v = np.concatenate([buildWordVector(z, n_dim) for z in tqdm(map(lambda x: x.words, x_test))])
test_vecs_w2v = scale(test_vecs_w2v) #scale each column to have zero mean and unit standard deviation.



from keras import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(32, activation='relu', input_dim=200))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(train_vecs_w2v, y_train, epochs=9, batch_size=32, verbose=2)



score = model.evaluate(test_vecs_w2v, y_test, batch_size=128, verbose=2)
print(score[1])

#next implementation : mode.evaluate with our own tweets


#new_data = postprocess(new_data)
#doit on le faire passer dans train test split ?
# x_new_train = labelizeTweets(x_new_train, 'PREDICTION') 

#vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=10) #ask to kevin
#When building the vocabulary ignore terms that have a document frequency strictly lower than the given threshold
#matrix2 = vectorizer.fit_transform([x.words for x in x_new_train])
#tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
# New_tweets = np.concatenate([buildWordVector(z, n_dim) for z in tqdm(map(lambda x: x.words, x_test))])
#New_tweets = scale(New_tweets)
#score = model.predict(New_tweets)

















#when you have dataframe, you have two do X.values To be sure that X is a numpy arrays values (scikit learn module need in input a numpy arrays values)
#to_categorical transform a vector in binary things
#if I want to save the model : 
"""
from keras.models import load_model
model.save('model_file.h5')
my_model = load_model('my_model.h5')

"""