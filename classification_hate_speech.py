#importing basic libraries
import pandas as pd
#reading dataset
data = pd.read_csv('final_dataset_basicmlmodel.csv')

#data cleaning
import re
def clean_text(text):
    # Filter to allow only alphabets

    text = re.sub(r'[^a-zA-Z\']', ' ', text)
    # Remove Unicode characters
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    # Convert to lowercase to maintain consistency
    text = text.lower()
    return text


data['clean_text'] = data.tweet.apply(lambda x: clean_text(x))

#feature engineering
#adding some user-defined stopwords to STOPWORDS

STOP_WORDS = ['a', 'about', 'above', 'after', 'again', 'against', 'all', 'also', 'am', 'an', 'and',
              'any', 'are', "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below',
              'between', 'both', 'but', 'by', 'can', "can't", 'cannot', 'com', 'could', "couldn't", 'did',
              "didn't", 'do', 'does', "doesn't", 'doing', "don't", 'down', 'during', 'each', 'else', 'ever',
              'few', 'for', 'from', 'further', 'get', 'had', "hadn't", 'has', "hasn't", 'have', "haven't", 'having',
              'he', "he'd", "he'll", "he's", 'her', 'here', "here's", 'hers', 'herself', 'him', 'himself', 'his', 'how',
              "how's", 'however', 'http', 'i', "i'd", "i'll", "i'm", "i've", 'if', 'in', 'into', 'is', "isn't", 'it',
              "it's", 'its', 'itself', 'just', 'k', "let's", 'like', 'me', 'more', 'most', "mustn't", 'my', 'myself',
              'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'otherwise', 'ought', 'our', 'ours',
              'ourselves', 'out', 'over', 'own', 'r', 'same', 'shall', "shan't", 'she', "she'd", "she'll", "she's",
              'should', "shouldn't", 'since', 'so', 'some', 'such', 'than', 'that', "that's", 'the', 'their', 'theirs',
              'them', 'themselves', 'then', 'there', "there's", 'these', 'they', "they'd", "they'll", "they're",
              "they've", 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', "wasn't",
              'we', "we'd", "we'll", "we're", "we've", 'were', "weren't", 'what', "what's", 'when', "when's", 'where',
              "where's", 'which', 'while', 'who', "who's", 'whom', 'why', "why's", 'with', "won't", 'would', "wouldn't",
              'www', 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves']


# Generate word frequency
def gen_freq(text):
    word_list = []
    for tw_words in text.split():
        word_list.extend(tw_words)
    word_freq = pd.Series(word_list).value_counts()
    word_freq = word_freq.drop(STOP_WORDS, errors='ignore')
    return word_freq


#Check whether a negation term is present in the text
def any_neg(words):
    for word in words:
        if word in ['n', 'no', 'non', 'not'] or re.search(r"\wn't", word):
            return 1
    else:
        return 0


#Check whether one of the 100 rare words is present in the text
def any_rare(words, rare_100):
    for word in words:
        if word in rare_100:
            return 1
    else:
        return 0


#Check whether prompt words are present
def is_question(words):
    for word in words:
        if word in ['when', 'what', 'how', 'why', 'who']:
            return 1
    else:
        return 0


word_freq = gen_freq(data.clean_text.str)
#100 most rare words in the dataset
rare_100 = word_freq[-100:]
#Number of words in a tweet
data['word_count'] = data.clean_text.str.split().apply(lambda x: len(x))
#Negation present or not
data['any_neg'] = data.clean_text.str.split().apply(lambda x: any_neg(x))
#Prompt present or not
data['is_question'] = data.clean_text.str.split().apply(lambda x: is_question(x))
#Any of the most 100 rare words present or not
data['any_rare'] = data.clean_text.str.split().apply(lambda x: any_rare(x, rare_100))
#Character count of the tweet
data['char_count'] = data.clean_text.apply(lambda x: len(x))


#train-test split
from sklearn.model_selection import train_test_split
X = data[['word_count', 'any_neg', 'any_rare', 'char_count', 'is_question']]
y = data.label
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=27)

#building and evaluating the model
from sklearn.naive_bayes import GaussianNB

#Initialize GaussianNB classifier
model = GaussianNB()
model = model.fit(X_train, y_train)
pred = model.predict(X_test)


from sklearn.metrics import accuracy_score
#model accuracy
print("Accuracy:", accuracy_score(y_test, pred)*100, "%")
#model accuracy is 60%
#though its very small, but it just for learning purpose and understanding how it works.