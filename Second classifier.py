
# coding: utf-8
#!/usr/bin/python2

"(C) Copyright 2018, Hesam Amoualian"

""""
# CNN classification using Keras for French sentiment analysis
"""




import sys
import numpy as np 
import pandas as pd 
import tensorflow
import keras
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import re
import json
import pandas as pd
from collections import Counter
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1,2,3"
import csv
import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import collections
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from imblearn.over_sampling import RandomOverSampler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import collections
from sklearn.metrics import recall_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint as sp_randint
from nltk.corpus import stopwords
from collections import defaultdict
from gensim import corpora
from gensim import corpora, models, similarities
import logging
from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
import nltk
import warnings
import numpy as np
import pandas as pd
import re
import itertools
from collections import Counter
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from keras.layers import Input, Dense, Embedding, Conv2D, MaxPool2D
from keras.layers import Reshape, Flatten, Dropout, Concatenate
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.models import Model
warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
config = tensorflow.ConfigProto( device_count = {'GPU': 3 , 'CPU': 3} ) 
sess = tensorflow.Session(config=config) 
keras.backend.set_session(sess)



def clean_str(string):
    string=string.replace(',',' ')
    string=string.replace('!',' ')
    string=string.replace('.',' ')
    string=string.replace('\'',' ')
    string = re.sub(r"[^A-Za-z(),!?\'\`èéêëôòóœàáâç]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " ", string)
    string = re.sub(r"!", " ", string)
    string = re.sub(r"\(", " ", string)
    string = re.sub(r"\)", " ", string)
    string = re.sub(r"\?", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string=string.replace('é','e')
    string=string.replace('è','e')
    string=string.replace('ê','e')
    string=string.replace('ç','c')
    string=string.replace('ć','c')
    string=string.replace('č','c')
    string=string.replace('ö','o')
    string=string.replace('ô','o')
    string=string.replace('ò','o')
    string=string.replace('ó','o')
    string=string.replace('á','a')
    string=string.replace('á','a')
    string=string.replace('â','a')
    newstring=[]
    for a in string.split():
        if len(a)>2:
            newstring.append(a)
            #newstring.append(nltk.stem.WordNetLemmatizer().lemmatize(a))
    string=' '.join(newstring)
    return string.strip()
    
def load_data_and_labels(filename):
    #df = pd.read_csv(filename,error_bad_lines=False,na_values=" ").fillna('nan')
    df = pd.read_csv(filename, encoding = 'UTF-16 LE',  sep = ';')
    data = df[['Tweets_data','Manual_Sentiment_Annotation']]
    # random_subset = data.sample(n=5000)
    # print(random_subset.head())
    # random_subset.to_csv('example.csv')
    data['sentiment']=['pos' if (x=='1') else 'neg' for x in data['Manual_Sentiment_Annotation']]
    data['Tweets_data']= [x.lower() for x in data['Tweets_data']]
    data['Tweets_data'] = data['Tweets_data'].apply((lambda x: re.sub('[^A-Za-z(),!?\'\`èéêëôòóœàáâç]',' ',x)))
    pd.set_option('display.max_colwidth',-1)
    data[:5]
    titles=data['Tweets_data'].values
    x_text = [clean_str(sent) for sent in titles]
    x_text = [s.split(" ") for s in x_text]
    y_input1=pd.get_dummies(data['sentiment']).values
    y_input=y_input1
    xnew=[]
    ynew=[]
    for n,a in enumerate(x_text):
        if len(a)>4 and len(a)<100:
            xnew.append(a)
            ynew.append(y_input[n])
    return [xnew, ynew]



def load_data_and_labels2(filename):
    df = pd.read_csv(filename,error_bad_lines=False,na_values=" ").fillna('nan')
    data = df[['polarity','statutnull']]
    # random_subset = data.sample(n=5000)
    # print(random_subset.head())
    # random_subset.to_csv('example.csv')
    data['sentiment']=['pos' if (x=='4') else 'neg' for x in data['polarity']]
    data['statutnull']= [x.lower() for x in data['statutnull']]
    data['statutnull'] = data['statutnull'].apply((lambda x: re.sub('[^A-Za-z(),!?\'\`èéêëôòóœàáâç]',' ',x)))
    pd.set_option('display.max_colwidth',-1)
    data[:5]
    titles=data['statutnull'].values
    x_text = [clean_str(sent) for sent in titles]
    x_text = [s.split(" ") for s in x_text]
    y_input1=pd.get_dummies(data['sentiment']).values
    y_input=y_input1
    xnew=[]
    ynew=[]
    for n,a in enumerate(x_text):
        if len(a)>4 and len(a)<100:
            xnew.append(a)
            ynew.append(y_input[n])
    return [xnew, ynew]


def pad_sentences(sentences, padding_word="<PAD/>"):
    sequence_length = max(len(x) for x in sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences
    
def build_vocab(sentences):
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    vocabulary_inv = list(sorted(vocabulary_inv))
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]
    
def build_input_data(sentences,labels, vocabulary):

    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    y = np.array(labels)
    return [x,y]
    
def load_data(filename):

    # Load and preprocess data
    sentences,labels = load_data_and_labels(filename)
    sentences_padded = pad_sentences(sentences)
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    x,y = build_input_data(sentences_padded,labels, vocabulary)
    return [x,y, vocabulary, vocabulary_inv]


def load_data2(filename):

    # Load and preprocess data
    sentences,labels = load_data_and_labels2(filename)
    sentences_padded = pad_sentences(sentences)
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    x,y = build_input_data(sentences_padded,labels, vocabulary)
    return [x,y, vocabulary, vocabulary_inv]


print('Loading data')
path2input = sys.argv[1]
print (path2input)
X,Y, vocabulary, vocabulary_inv = load_data2(path2input)
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 42)
X2,Y2, vocabulary2, vocabulary_inv2 = load_data("vire2.csv")
X2_train, X2_test, Y2_train, Y2_test = train_test_split(X2,Y2, test_size = 0.2, random_state = 42)
sequence_length = X.shape[1]
vocabulary_size = len(vocabulary_inv)
embedding_dim = 300
filter_sizes = [1,2,3,4,5,6]
num_filters = 512
drop = 0.5

epochs = 20
batch_size = 30

print("Creating Model...")
inputs = Input(shape=(sequence_length,), dtype='int32')
embedding = Embedding(input_dim=vocabulary_size, output_dim=embedding_dim, input_length=sequence_length)(inputs)
reshape = Reshape((sequence_length,embedding_dim,1))(embedding)

conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
conv_3 = Conv2D(num_filters, kernel_size=(filter_sizes[3], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
conv_4 = Conv2D(num_filters, kernel_size=(filter_sizes[4], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
conv_5 = Conv2D(num_filters, kernel_size=(filter_sizes[5], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
maxpool_0 = MaxPool2D(pool_size=(sequence_length - filter_sizes[0] + 1, 1), strides=(1,1), padding='valid')(conv_0)
maxpool_1 = MaxPool2D(pool_size=(sequence_length - filter_sizes[1] + 1, 1), strides=(1,1), padding='valid')(conv_1)
maxpool_2 = MaxPool2D(pool_size=(sequence_length - filter_sizes[2] + 1, 1), strides=(1,1), padding='valid')(conv_2)
maxpool_3 = MaxPool2D(pool_size=(sequence_length - filter_sizes[3] + 1, 1), strides=(1,1), padding='valid')(conv_3)
maxpool_4 = MaxPool2D(pool_size=(sequence_length - filter_sizes[4] + 1, 1), strides=(1,1), padding='valid')(conv_4)
maxpool_5 = MaxPool2D(pool_size=(sequence_length - filter_sizes[5] + 1, 1), strides=(1,1), padding='valid')(conv_5)
concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2, maxpool_3, maxpool_4,maxpool_5])
flatten = Flatten()(concatenated_tensor)
dropout = Dropout(drop)(flatten)
output = Dense(2, activation='softmax')(dropout)

model = Model(inputs=inputs, outputs=output)

checkpoint = ModelCheckpoint('weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
print("Traning Model...")
model.fit(X2_train, Y2_train, batch_size=batch_size, epochs=epochs, verbose=1, callbacks=[checkpoint], validation_data=(X_test, Y_test))  # starts training
score = model.evaluate(X_test, Y_test, batch_size=batch_size, verbose=1)
print(score[1])

# model_json = model.to_json()
# with open("model_in_json.json", "w") as json_file:
#     json.dump(model_json, json_file)

# model.save_weights("model_weightsbruh.h5")