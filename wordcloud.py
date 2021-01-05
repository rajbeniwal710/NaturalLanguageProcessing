#Load the dataset
import pandas as pd
dataset = pd.read_csv('tweets.csv', encoding = 'ISO-8859-1')
#generating word frequency
#generating a frequency table of all the words present in all the tweets combined
def gen_freq(text):
    # Will store the list of words
    word_list = []
    # Loop over all the tweets and extract words into word_list
    for tw_words in text.split():
        word_list.extend(tw_words)
    # Create word frequencies using word_list
    word_freq = pd.Series(word_list).value_counts()
    # Print top 20 words
    word_freq[:20]
    return word_freq

gen_freq(dataset.text.str)

#EDA using WordCloud
import matplotlib.pyplot as plt
from wordcloud import WordCloud

#Generate word frequencies
word_freq = gen_freq(dataset.text.str)
def word_cloud():
    #Generate word cloud
    wc = WordCloud(width=400, height=330, max_words=100, background_color='white').generate_from_frequencies(word_freq)
    plt.figure(figsize=(12, 8))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.show()
word_cloud()
#text cleaning using regex

import re

def clean_text(text):
    # Remove RT
    text = re.sub(r'RT', '', text)
    # Fix &
    text = re.sub(r'&amp;', '&', text)
    # Remove punctuations
    text = re.sub(r'[?!.;:,#@-]', '', text)
    # Convert to lowercase to maintain consistency
    text = text.lower()
    return text

#Import list of stopwards
from wordcloud import STOPWORDS

text = dataset.text.apply(lambda x: clean_text(x))
word_freq = gen_freq(text.str)*100
word_freq = word_freq.drop(labels=STOPWORDS, errors='ignore')

word_cloud()