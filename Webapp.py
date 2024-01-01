from xml.dom import NotFoundErr
import tweepy
import csv
import pandas as pd
from PIL import Image
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import tree
import tweepy
import csv
import re
import spacy
spacy.cli.download("en_core_web_lg")

import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
import time
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
import contractions
from sklearn import tree
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import itertools
import string
import pickle
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


st.write("""
# Depression Detection
Detect if some twitter user has depression using machine learning and python""")
image = Image.open('image.jpg')
st.image(image, caption='ML', use_column_width=True)

#vectorizer = TfidfVectorizer()
svc = pickle.load(open('svm1.pkl', 'rb'))   

def get_all_tweets(screen_name):  
    consumer_key = "FHSCcqycpgpHoFZ1OqZKtNLKE"
    consumer_secret = "YNdtiBJXuyMuTP0QyfAoEGbFvizyoIjCPZeUgDAwLqB2kJnOhc"
    access_key = "1501899564038881287-Mrd6cNZqIXYQGOxE0iRDPrW6oyhSbp"
    access_secret = "XoLZwCuVkogoxLZnL1Llxvc8GjJDIoykuxHX12YjZq20o"
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tweepy.API(auth, wait_on_rate_limit=True)
    alltweets = []
    noRT = []
    new_tweets = api.user_timeline(screen_name = screen_name, tweet_mode = 'extended', count=200)
    alltweets.extend(new_tweets)
    oldest = alltweets[-1].id - 1
    while len(new_tweets) >0:
        print("getting tweets before {}".format(oldest))
        new_tweets = api.user_timeline(screen_name = screen_name,tweet_mode = 'extended', count=200,max_id=oldest)
        alltweets.extend(new_tweets)
        oldest = alltweets[-1].id - 1
        print("...{} tweets downloaded so far".format(len(alltweets)))
    for tweet in alltweets:
        if ('RT' in tweet.full_text or '@' in tweet.full_text):
            continue
        else:
            noRT.append([tweet.id_str, tweet.created_at, tweet.full_text])
    with open('{}_tweets.csv'.format(screen_name), 'w', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id","created_at","text"])
        writer.writerows(noRT)
        user=pd.read_csv('{}_tweets.csv'.format(screen_name))
        st.subheader('Data Information:')
        st.dataframe(user)
    pass


def word_is_negated(word):
    for child in word.children:
        if child.dep_ == 'neg':
            return True
    if word.pos_ in {'VERB'}:
        for ancestor in word.ancestors:
            if ancestor.pos_ in {'VERB'}:
                for child2 in ancestor.children:
                    if child2.dep_ == 'neg':
                        return True
    return False


def find_negated_wordSentIdxs_in_sent(sent, idxs_of_interest=None):
    negated_word_idxs = set()
    for word_sent_idx, word in enumerate(sent):
        if idxs_of_interest:
            if word_sent_idx not in idxs_of_interest:
                continue
        if word_is_negated(word):
            negated_word_idxs.add(word_sent_idx)
    return negated_word_idxs 



def pred(inputtweet):
    tweet = pd.read_csv('{}_tweets.csv'.format(inputtweet))
    tweet["label"] = ""
    tweet.drop(['created_at'], axis = 1, inplace = True)
    tweet.drop(['id'], axis = 1, inplace = True)
    i=0
    nlp = spacy.load('en_core_web_lg')
    for row in tweet['text']:
        RE_EMOJI = re.compile('[\U00010000-\U0010ffff]', flags=re.UNICODE)
        row = RE_EMOJI.sub(r'', row)
        row = ''.join([c for c in row if ord(c) < 128])
        row=contractions.fix(row)
    
    
        j=find_negated_wordSentIdxs_in_sent(nlp(row))
        row=re.sub(r"http\S+|www\S+|https\S+","",row,flags=re.MULTILINE)
        row=re.sub(r'\@\w+|\#',"",row,flags=re.MULTILINE)
        row=row.translate(str.maketrans("","",string.punctuation))
        row = ''.join([c for c in row if ord(c) < 128])
        row=row.strip()
        row_tokens=word_tokenize(row)
        filtered_words=[word for word in row_tokens if word not in stopwords.words('english')]
        ps=PorterStemmer()
        stemmed_words=[ ps.stem(w) for w in filtered_words]
        lemmatizer=WordNetLemmatizer()
        lemma_words=[lemmatizer.lemmatize(w,pos='a') for w in stemmed_words]
        row = " ".join(lemma_words)
        vectorizer = pickle.load(open('vectorizersvc.pickle', 'rb'))
        inputdtree= vectorizer.transform([row])
        predictt = svc.predict(inputdtree)


        
        if (j != set()):
            tweet.loc[i,'label'] = int(not(predictt))
        else:
            tweet.loc[i,'label'] = int(predictt)
        i=i+1
    nodep=(tweet.label == 0).sum()
    dep=(tweet.label == 1).sum()
    sum=nodep+dep
    percentage=dep/sum
    st.subheader('Depression Level:')
    if (percentage>=0 and percentage<=0.25):
        st.write("Considered Normal")
    elif (percentage>=0.25 and percentage<=0.40):
        st.write("Mild Depression")
    else:
        st.write("Severe Depression")
        st.video("DealingDepression.mp4")
    print (dep)
    print (percentage)

    st.subheader('Training Data Information:')
    st.write("Available [here](https://drive.google.com/file/d/1QK5IOH4mWeFtqppTBbxDuEonohhrjdIj/view?usp=sharing)")
 
    st.subheader('WordCloud Analysis of Training Data:')
    st.write("Available [here](https://drive.google.com/drive/folders/1VMG0MgJ-nZrEYRTLBxbKVwtwNVUKXq0N?usp=sharing)")

    st.subheader('Model Metrics: ')
    st.write("Available [here](https://drive.google.com/file/d/1EokLG36SlHz_HSBDdVnFE-NkzWS64_p8/view?usp=sharing)")
try:
    with st.form(key='my_form'):
        inputtweet = st.text_input(label='Input your twitter handle without @:')
        submit_button = st.form_submit_button(label='Check')
        get_all_tweets(inputtweet)

    
    pred(inputtweet)
except:
    time.sleep(10)
    st.info("Waiting for your correct input...")
    
