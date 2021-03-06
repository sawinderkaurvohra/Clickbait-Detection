# Clickbait-Detector
Detects clickbait using two-phase CNN-LSTM Bi-term model. At first phase, hybrid CNN-LSTM model is used to accurately detect the fed short text as clickbait or not. At second phase, Biterm model is combined with the clickbait headline retrieved from the previous phase to classify the type of clickbait (reasoning, number, reaction, revealing, shocking/unbelievable, hypothesis/guess, questionable, forward referencing).

##  [Check demo](https://clickbait.isitfake.co.in)
Django based web app hosted on AWS.

# Table of contents
1) [Jupyter Notebook code implementation](/notebook/Clickbaitness_Analysis_with_BTM_(CNN+LSTM_model).ipynb)
2) [Python code](/code/cnn_lstm+btm.py)
3) [Datasets](/dataset/)
4) [Model file](/model/model_cnn_lstm.h5)

# Requirements
* Python 3.6.3
* Keras 2.3.1
* Tensorflow 2.0.0
* Numpy 1.17.4
* Scipy 1.3.2
* Django 2.2.7 (web app)

# Get started
1. Clone this repo
   `git clone https://github.com/sawinderkaurvohra/Clickbait-Detector`
2. Setup virtual environment using pipenv package.
   `pipenv shell`
3. Install dependencies.
   `pipenv install -r requirements.txt`
4. Start script
   `cd code && python3 cnn_lstm+btm.py`

# Testing of clickbait and non-clickbait samples on web app
1. Clickbait headlines
   * Textual

   <img src="images/clickbait-output.png" width="370px" height="250px"/>

   * Non-Textual

   <img src="images/clickbait_image_output1.png" width="370px" height="680px"/>

1. Non-Clickbait headlines
   * Textual

   <img src="images/not-clickbait-output.png" width="370px" height="250px"/>
  
   * Non-Textual

   <img src="images/non_clickbait_image_output.png" width="370px" height="500px"/>



# Model summary
![Model Summary](/images/modelsummary.png)

# Performance analysis
   1. OCR performance: Google Cloud Vision tool was used to extract headlines from the non-textual (in form of images) posts of our collected corpus. A post-processing step was performed on the tool to improve the accuracy of the API. It was observed that the accuracy improved by 1.34%. Operations used for pre-processing were converting image to grayscale, noise removal, extraction of headlines using Gooogle API and auto-correction of the textual data retrieved from the image.
   1. Classification performance: Glove pre-trained vectors when embedded with hybrid CNN-LSTM model gave best accuracy on all the three collected corpus. The model achieved an accuracy of 95.8%, 89.44%, 94.21% with 0.99, 0.94, 0.98 ROC-AUC values for Dataset 1, Dataset 2 and Ground Dataset, respectively.


# Glove pre-trained embedding
The vector size chosen for word embeddings is 100-dimensional in our experiment. Python library named as glove python is used to implement Glove embeddings. Glove embedding used in this research has 6B tokens, 400K vocab, 100d vectors.
[Download embeddings](http://nlp.stanford.edu/data/glove.6B.zip)


# Biterm Topic Model (BTM)
BTM is a word co-occurrence bi-term based model as where a bi-term consists of two words co-occurrence in the same headline. It is a type of generative model which generates a bi-term by making a two word pattern from the same topic.

# Cluster analysis
Elbow method is is designed to find out the appropriate number of clusters in dataset. The number of cluster where a bend is seen in the curve (the point where the marginal gain is dropped), that is chosen to get better modelling of data. The value of t (no. of topics) chosen to apply BTM in our experiment was 9.
![Cluster analysis](/images/main-cluster.png)

# Corpus
The dataset was collected from two sources, i.e., Reddit using web-scraping tool Octoparse and Facebook (non-textual data in form of images) using human annotations. The dataset was collected to analyze the distribution of both clickbait and non-clickbait headlines in terms of shares, likes, comments, domains, time, etc (available from 1-DEC-2016 to 21-JUN2019). The non-clickbait headlines were collected from subreddits, which do not allow the clickbaits to creep in. 

The other two corpus used for training the two-phase model are having 12,000 and 32,000 clickbait and non-clickbait headlines. The statistics of dataset used is:

Corpus        | Total headlines | Clickbait headlines  | Non-clickbait headlines |    
------------- | -------------   | -------------------- | ----------------------- |   
Dataset 1     | 32,000          | 15,999               | 16,001                  |
Dataset 2     | 12,000          | 5,637                | 6,080                   |
Ground Dataset     | 1,800           | 1,200                | 600                     |

# Screenshots

* Dashboard
   <img src="images/dashboard.png"/>

* Text Input Form
   <img src="images/sc_2.png"/>

* Output of Text Input
   <img src="images/sc_3.png"/>

* Graph Analysis of Input Text
   <img src="images/sc_4.png"/>

* Image Input Form
   <img src="images/sc_5.png"/>
  
* Output of Image Input
   <img src="images/sc_6.png"/>
   <img src="images/sc_7.png"/>
