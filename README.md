# Clickbait-Detector
Detects clickbait using two-phase CNN-LSTM Bi-term model. At first phase, hybrid CNN-LSTM model is used to accurately detect the fed short text as clickbait or not. At second phase, Biterm model is combined with the clickbait headline retrieved from the previous phase to classigy the type of clickbait (reasoning, number, reaction, revealing, shocking/unbelievable, hypothesis/guess, questionable, forward referencing)
# Requirements
* Python 3.6
* Keras 2.3.1
* Tensorflow 2.0.0
* Numpy 1.17.4
* scipy 1.3.2
* Django 2.2.7
# Get started

# Model summary
![GitHub Logo](/images/logo.png)
Format: ![Alt Text](url)
# Performnace analysis


# Glove pre-trained embedding
The vector size chosen for word embeddings is 100-dimensional in our experiment. Python library named as glove python is used to implement Glove embeddings.


# Bi-term model
BTM is a word co-occurrence bi-term based model as where a bi-term consists of two words co-occurrence in the same headline. It is a type of generative model which generates a bi-term by making a two word pattern from the same topic.
# Cluster analysis

# Corpus
The dataset was collected from two sources, i.e., Reddit using web-scraping tool Octoparse and Facebook (non-textual data in form of images) using human annotations. The dataset was collected to analyze the distribution of both clickbait and non-clickbait headlines in terms of shares, likes, comments, domains, time, etc (available from 1-DEC-2016 to 21-JUN2019). The non-clickbait headlines were collected from subreddits, which do not allow the clickbaits to creep in. 

The other two corpus used for training the two-phase model are having 12000 and 31000 clickbait and non-clickbait headlines. The statistics of dataset used is
Corpus        | Total headlines | Clickbait headlines  | Non-clickbait headlines |     
------------- | -------------   | -------------------- | ----------------------- |   
Dataset 1     | 32,000          | 15,999               | 16,001                  |
Dataset 2     | 12,000          | 5,637                | 6,080                   |
Dataset 3     | 1,800           | 1,200                | 600                     |
