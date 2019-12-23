#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[29]:


from google.cloud import vision
from googleapiclient.discovery import build
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np
import pyLDAvis
from biterm.btm import oBTM
from sklearn.feature_extraction.text import CountVectorizer
from biterm.utility import vec_to_biterms, topic_summuary
from IPython.display import IFrame

import os
import io
import numpy as np


# # Define variables

# In[33]:


MAX_VOCAB_SIZE = 20000
MAX_SEQUENCE_LENGTH = 100

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "../assets/vision.json"
SERVICE = build("customsearch", "v1",
                developerKey="AIzaSyAaJ8-PEOVH4AiNEZ2KcS24h48tPIkrmdY")
image_path = "../assets/verified.jpg"
model = load_model('../assets/host.h5')
DATASET = "../dataset/32000.csv"


# # Start clickbait from Text (Test)

# In[34]:


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

# In[36]:


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


# # classify mass data 

# In[ ]:


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
pyLDAvis.save_html(vis, 'BTM.html')

print("\n\n Topic coherence ..")
topic_summuary(btm.phi_wz.T, X, vocab, 10)

print("\n\n Texts & Topics ..")
for i in range(len(texts)):
    print("{} (topic: {})".format(texts[i], topics[i].argmax()))

