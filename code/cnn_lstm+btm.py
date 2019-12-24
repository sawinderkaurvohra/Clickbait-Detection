#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[15]:


from __future__ import print_function, division
from biterm.btm import oBTM
from sklearn.feature_extraction.text import CountVectorizer
from biterm.utility import vec_to_biterms, topic_summuary
from google.cloud import vision
from googleapiclient.discovery import build
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from sklearn.metrics import roc_auc_score
from keras.models import model_from_json
from keras.layers import Dense, Embedding, LSTM, GRU
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.models import load_model
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from builtins import range
from IPython.display import IFrame

import os
import sys
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import pickle
import json
import pyLDAvis


# # Define variables

# In[2]:


MAX_VOCAB_SIZE = 20000
MAX_SEQUENCE_LENGTH = 100

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "../assets/vision.json"
SERVICE = build("customsearch", "v1",
                developerKey="AIzaSyAaJ8-PEOVH4AiNEZ2KcS24h48tPIkrmdY")
image_path = "../assets/verified.jpg"
model = load_model('../assets/host.h5')

EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 128
EPOCHS = 15
GLOVE_DIR = "emb"

DATASET = "../dataset/32000.csv"


# # CNN + LSTM

# ## load in pre-trained word vectors

# In[3]:


# load in pre-trained word vectors
print('Loading word vectors...')
GLOVE_DIR = "../emb"
word2vec = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'), encoding="utf8")
#with line as f:
  # is just a space-separated text file in the format:
  # word vec[0] vec[1] vec[2] ...
for line in f:
    values = line.split()
    word = values[0]
    vec = np.asarray(values[1:], dtype='float32')
    word2vec[word] = vec
print('Found %s word vectors.' % len(word2vec))


# ## prepare text samples and their labels

# In[4]:


print('Loading in comments...')

train = pd.read_csv(DATASET)
sentences = train["text"].fillna("DUMMY_VALUE").values
tt="hi"
possible_labels = ["label"]
targets = train[possible_labels].values


# ## convert the sentences (strings) into integers

# In[5]:


tt=np.array(['come on'])
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)

tokenizer2 = Tokenizer(num_words=MAX_VOCAB_SIZE)
tokenizer2.fit_on_texts(tt)
sequences2 = tokenizer2.texts_to_sequences(tt)
print("sequences:", sequences[0][0])


#print("max sequence length:", max(len(s) for s in sequences))
#print("min sequence length:", min(len(s) for s in sequences))
s = sorted(len(s) for s in sequences)
#print("median sequence length:", s[len(s) // 2])


# get word -> integer mapping
word2idx = tokenizer.word_index
#print('Found %s unique tokens.' % len(word2idx))


# pad sequences so that we get a N x T matrix
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
data2 = pad_sequences(sequences2, maxlen=MAX_SEQUENCE_LENGTH)


# ## prepare embedding matrix

# In[6]:


print('Filling pre-trained embeddings...')
num_words = min(MAX_VOCAB_SIZE, len(word2idx) + 1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word2idx.items():
    if i < MAX_VOCAB_SIZE:
        embedding_vector = word2vec.get(word)
        if embedding_vector is not None:
          # words not found in embedding index will be all zeros.
          embedding_matrix[i] = embedding_vector



# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
embedding_layer = Embedding(
  num_words,
  EMBEDDING_DIM,
  weights=[embedding_matrix],
  input_length=MAX_SEQUENCE_LENGTH,
  trainable=False
)


print('Building model...')

# train a 1D convnet with global maxpooling
input_ = Input(shape=(MAX_SEQUENCE_LENGTH,))
x = embedding_layer(input_)

x = Conv1D(128, 3, activation='relu')(x)
x = MaxPooling1D(3)(x)
x = Conv1D(128, 3, activation='relu')(x)
x = MaxPooling1D(3)(x)
x = Conv1D(128, 3, activation='relu')(x)
#x = GlobalMaxPooling1D()(x)
#x = Dense(128, activation='relu')(x)
x = LSTM(100, dropout=0.2, recurrent_dropout=0.2)(x)
#x = BatchNormalization(x)
x = Dense(256, activation='relu')(x)
# x = Dense(128, activation='relu')(x)
# x = Dense(64, activation='relu')(x)
# x = Dense(32, activation='relu')(x)
# modell.add(Dropout(0.2))
# modell.add(Dense(32, activation='relu'))
x = Dense(2, activation='softmax')(x)
output = Dense(len(possible_labels), activation='sigmoid')(x)
model = Model(input_, output)
print(model.summary())

model.compile(
  loss='binary_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)

print('Training model...')
r = model.fit(
  data,
  targets,
  batch_size=BATCH_SIZE,
  epochs=EPOCHS,
  validation_split=VALIDATION_SPLIT
)


# ## Plot Graph

# In[7]:


plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')

plt.legend()
plt.show()

# accuracies
plt.plot(r.history['accuracy'], label='accuracy')
plt.plot(r.history['val_accuracy'], label='val_accuracy')
plt.legend()
plt.show()

# plot the mean AUC over each label
p = model.predict(data)
#aucs = []
#for j in range(6):
 #   auc = roc_auc_score(targets[:,j], p[:,j])
  #  aucs.append(auc)
#print(np.mean(aucs))


# ## Save Model

# In[8]:


try:
    # serialize model to JSON
    model_json = model.to_json()
    with open("modeln.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("modeln.h5")
    print("Saved model to disk")

except:
    # serialize model to JSON
    model_json = r.to_json()
    with open("modrel.json", "w") as json_file:
        json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights("mordel.h5")
    print("Saved model to disk")

ynew = model.predict(data2)

print(ynew[0])


# # Predict clickbait from Text (Test)

# In[12]:


def check_clickbaitness(text):
    input_text=np.array([text])
    tokenizer_predict = Tokenizer(num_words=MAX_VOCAB_SIZE)
    tokenizer_predict.fit_on_texts(input_text)
    sequences = tokenizer_predict.texts_to_sequences(input_text)
    data_predict = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    return model.predict(data_predict)[0][0]

score = check_clickbaitness('Floating wreckage of Brazilian plane carrying four UK businessmen recovered')
print('Score is %.5f' % (score))
print('Text is ' + ('clickbait' if score >= 0.5 else 'not a clickbait'))


# # Image to Text for clickbaitness

# In[13]:


def text_detection(path):
    client = vision.ImageAnnotatorClient()
    with io.open(path, 'rb') as image_file:
        content = image_file.read()
    image = vision.types.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations
    for text in texts:
        return text.description
fetched = text_detection(image_path)
print('Text is ' + fetched)
score = check_clickbaitness(fetched)
print('Score is %.5f' % (score))
print('Text is ' + ('clickbait' if score >= 0.5 else 'not a clickbait'))


# # Classify mass data 

# In[14]:


df = pd.read_csv('../dataset/32000.csv')['text']

clickbait = []
non_clickbait = []
for text in df:
    score = check_clickbaitness(text)
    clickbait.append(text) if score >= 0.5 else non_clickbait.append(text)

df = pd.DataFrame(clickbait)
df.to_csv('clickbait.csv')

df = pd.DataFrame(non_clickbait)
df.to_csv('non_clickbait.csv')


# # Find cluster occurence using BTM

# In[24]:


texts = list(pd.read_csv('../dataset/Click_&_non_click_1800.csv', encoding = 'ISO-8859-1')['text'])[200:700]
# vectorize texts
vec = CountVectorizer(stop_words='english')
X = vec.fit_transform(texts).toarray()

# get vocabulary
vocab = np.array(vec.get_feature_names())

# get biterms
biterms = vec_to_biterms(X)

# create btm
btm = oBTM(num_topics=9, V=vocab)

print("\n\n Train BTM ..")
topics = btm.fit_transform(biterms, iterations=100)

print("\n\n Visualize Topics ..")
vis = pyLDAvis.prepare(btm.phi_wz.T, topics, np.count_nonzero(X, axis=1), vocab, np.sum(X, axis=0))
pyLDAvis.save_html(vis, '../assets/BTM.html')

print("\n\n Topic coherence ..")
topic_summuary(btm.phi_wz.T, X, vocab, 10)

print("\n\n Texts & Topics ..")
for i in range(len(texts)):
    print("{} (topic: {})".format(texts[i], topics[i].argmax()))


# In[16]:


IFrame(src='../assets/BTM.html', width=700, height=600)

