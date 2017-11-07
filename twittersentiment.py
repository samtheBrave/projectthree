from __future__ import print_function
from allfilelist import allfilelist
import pandas as pd
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from nltk.tokenize import TweetTokenizer
from TweetPreprocessor import TweetPreprocessor
from keras.utils import to_categorical


import numpy as np
import tensorflow as tf



max_features = 20000
maxlen = 80  # cut texts after this number of words (among top max_features most common words)
batch_size = 32



def cleantweets(tweets):
    tweet_processor = TweetPreprocessor()

    tknzr = TweetTokenizer()


    docs = [tknzr.tokenize(tweet_processor.preprocess(tweet, 'Twitter')) for  tweet in
            (tweets)]

    return docs




pathroot = 'D:/Dropbox/1. Raw Data_2017_IOT/IOT iteration done/IOT 2017 Twitter/'

fileall = allfilelist.allfilesinlist(allfilelist(), pathroot, 'csv')

list_ = []

for file in fileall[:]:

    #print(file)
    tempdf = pd.read_csv(file,index_col=None, header=0, encoding="ISO-8859-1", low_memory=False,error_bad_lines=False,skip_blank_lines=True)
    list_.append(tempdf)


alltweets = pd.concat(list_)

alltweets = alltweets.fillna('Neutral')

alltweets.Emotion = pd.Categorical(alltweets.Emotion)
alltweets['Emotype'] =alltweets.Emotion.cat.codes


labels =(alltweets['Emotype'])
processedtweets = cleantweets(alltweets['Contents'])

words = [val for sublist in processedtweets for val in sublist]




from collections import Counter
counts = Counter(words)
vocab = sorted(counts, key=counts.get, reverse=True)
vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}




reviews_ints = []

for each in processedtweets:

    reviews_ints.append([vocab_to_int[word] for word in each])



labels = np.array([each for each in labels])
print(labels)
review_lens = Counter([len(x) for x in reviews_ints])
print("Zero-length reviews: {}".format(review_lens[0]))
print("Maximum review length: {}".format(max(review_lens)))
non_zero_idx = [ii for ii, review in enumerate(reviews_ints) if len(review) != 0]

reviews_ints = [reviews_ints[ii] for ii in non_zero_idx]
labels = np.array([labels[ii] for ii in non_zero_idx])
labels = to_categorical(labels)
seq_len = 200
features = np.zeros((len(reviews_ints), seq_len), dtype=int)

for i, row in enumerate(reviews_ints):
    features[i, -len(row):] = np.array(row)[:seq_len]

split_frac = 0.8
split_idx = int(len(features)*0.8)
x_train, x_val = features[:split_idx], features[split_idx:]
y_train, y_val = labels[:split_idx], labels[split_idx:]


test_idx = int(len(x_val)*0.5)
x_val, x_test  = x_val[:test_idx], x_val[test_idx:]
y_val, y_test = y_val[:test_idx], y_val[test_idx:]
print(x_train[1])
print(len(x_train[1]))
print(y_train[1])


print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(7, activation='softmax'))

# try using different optimizers and different optimizer configs
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=15,
          validation_data=(x_test, y_test))
score, acc = model.evaluate(x_test, y_test,
                            batch_size=batch_size)


print('Test score:', score)
print('Test accuracy:', acc)


model.save('tweeter_emotions.h5')