import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import pandas as pd

labels = ["Analyst Update", "Fed | Central Banks", "Company | Product News", "Treasuries | Corporate Debt", "Dividend", "Earnings", "Energy | Oil", "Financials", "Currencies", "General News | Opinion", "Gold | Metals | Materials", "IPO", "Legal | Regulation", "M&A | Investments", "Macro", "Markets", "Politics", "Personnel Change", "Stock Commentary", "Stock Movement"]
stopword=set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text=" ".join(text)
    # text = [stemmer.stem(word) for word in text.split(' ')]
    text = [lemmatizer.lemmatize(word) for word in text.split(' ')]
    text=" ".join(text)
    return text

# get the tweet in this function and return the financial classification of the tweet
def classify(tweet):
    """machine learning function which accepts tweet in string format

    Keyword arguments:
    tweet -- string of tweet
    Return: tweet classification
    """

    # load the CountVectorizer & RandomForest from pkl file
    cv = pickle.load(open('model/cv.pkl', 'rb'))
    rf = pickle.load(open('model/rf.pkl', 'rb'))

    # vectorize tweet & use random forest to classify
    tweet_df = pd.DataFrame({'text':[tweet]})
    tweet_df['text'] = tweet_df['text'].apply(clean)
    cvx = cv.transform(tweet_df['text'])
    result = rf.predict(cvx)
    print(result[0])
    return labels[result[0]]