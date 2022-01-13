import re
import string
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import RegexpTokenizer

def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~`" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'

def clean_special_chars(text, punct):
    for p in punct:
        text = text.replace(p, ' ')
    return text

def process_tweet(tweet):
    """Process tweet function.
    Input:
        tweet: a string containing a tweet
    Output:
        tweets_clean: a list of words containing the processed tweet
    """
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()    
    stopwords_english = stopwords.words('english')
    # remove stock market tickers like $GE
    tweet = re.sub(r'\$\w*', '', tweet)
    # remove old style retweet text "RT"
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    # remove hyperlinks
    tweet = re.sub(r'http\S+', '', tweet)
    # remove hashtags
    # only removing the hash #, @, ... sign from the word
    tweet = re.sub(r'\.{3}|@|#', '', tweet)    
    tweet = decontracted(tweet)    
    # remove junk characters which don't have an ascii code
    tweet = tweet.encode("ascii", "ignore").decode("utf-8", "ignore")
    # tokenize tweets    
    tweet = clean_special_chars(tweet, punct)
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)
    tweets_clean = []
    for word in tweet_tokens:
        print(word)
        # remove stopwords and punctuation
        if (word.isalpha() and word not in stopwords_english and word not in string.punctuation):
            stem_word = stemmer.stem(word)  # stemming word
            #lemma_word = lemmatizer.lemmatize(word)
            tweets_clean.append(stem_word)            
    return tweets_clean

def add_or_increment(key, dict):
    if key in dict:
        dict[key] += 1
    else:
        dict[key] = 1       

def build_freqs(tweets, ys):
    """Build frequencies.
    Input:
        tweets: a list of tweets
        ys: an m x 1 array with the sentiment label of each tweet (either 0 or 1)
    Output:
        freqs: a dictionary mapping each (word, sentiment) pair to its frequency
    """
    # Convert np array to list since zip needs an iterable.
    # The squeeze is necessary or the list ends up with one element.
    # Also note that this is just a NOP if ys is already a list.
    yslist = np.squeeze(ys).tolist()

    # Start with an empty dictionary and populate it by looping over all tweets
    # and over all processed words in each tweet.
    freqs = {}
    pos_freqs = {}
    neg_freqs = {}
    for y, tweet in zip(yslist, tweets):
        for word in process_tweet(tweet):
            pair = (word, y)
            add_or_increment(pair, freqs)
            add_or_increment(word, pos_freqs)
            add_or_increment(word, neg_freqs)
    return freqs, pos_freqs, neg_freqs

def get_topn_dictitems_byvalue(dict_data, n):
    freq_word = [[int(value), key] for key, value in dict_data.items()]
    freq_word.sort(reverse=True, key=lambda k: k[0])
    return freq_word[:n]
