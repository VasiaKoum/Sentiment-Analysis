from string import punctuation
from nltk.corpus import stopwords
from collections import Counter
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer 
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyser = SentimentIntensityAnalyzer()
def sentiment_analyzer_scores(sentence):
    return analyser.polarity_scores(sentence)['compound']

def clean_tweets(filepath, emoticons):
    text = open(filepath, 'r', encoding="utf8").read() 
    tweets_list = text.splitlines() # split the tweets in lines
    # emoticons dictionary
    all_tweets = []
    for tweet in tweets_list:  # for each tweet
        cur_tweet = []
        words = tweet.split() # split the tweet into a list of words
        reformed = [emoticons[word] if word in emoticons else word for word in words]   # replace every emoticon in the tweet(if any) with its word mapping
        new_tweet = " ".join(reformed)  # reconstruct the tweet
        words = new_tweet.split() # split the new tweet into a list of words 
        for word in words[3:]:  # actual tweet starts at index 3
            if (word[0] == '@') or (word[0] == '#') or (word[:4] == 'http'): # ignore @somename, #somehashtag and http(s)://somelink
                continue
            clean_word = word.strip(punctuation).lower() # remove the special characters and then convert it to lowercase 
            if clean_word == '': # if the cleaned string is the empty string ignore it as well
                continue
            if clean_word in stopwords.words('english'): # ignore the stopwords
                continue
            #all_words.append(clean_word) # add the cleaned word to the list containing all the words
            cur_tweet.append(clean_word) # add each cleaned word of the current tweet to a list
        cleaned_tweet = " ".join(cur_tweet) # cleaned tweet reconstructed, using the cur_tweet list of words
        all_tweets.append(cleaned_tweet)    # add the cleaned tweet to the list of all tweets
    return all_tweets


def get_labels(filepath):
    text = open(filepath, 'r', encoding="utf8").read() 
    tweets_list = text.splitlines()     # split the tweets in lines
    labels = []
    for tweet in tweets_list:
        words = tweet.split()
        if len(words) == 2: # *gold.txt
            labels.append(words[1])
        else:
            labels.append(words[2])

    return labels

def tweet2vector(w2vmodel, tokens_list, num_of_features, lexicon_1, lexicon_2):
    vector = np.zeros(num_of_features).reshape( (1, num_of_features) ) # create a vector of num_of_features features and fill it with 0's
    count = 0
    lex_1 = 0
    lex_1c = 0
    lex_2 = 0
    lex_2c = 0
    max_valence = -1000
    min_valence = 1000

    if len(tokens_list) == 0:         # tweet is empty after the cleaning - happens for some tweets
        vector = w2vmodel['tomorrow'].reshape( (1, num_of_features) ) # assign to the tweet's vector, a vector of a neutral/very common word
                                                                      # 'tomorrow' seems to give good results
    for word in tokens_list:
        if lexicon_1 is not None:
            if word in lexicon_1:
                lex_1 += lexicon_1[word]
                lex_1c += 1
        if lexicon_2 is not None:
            if word in lexicon_2:
                lex_2 += lexicon_2[word]
                lex_2c += 1
        try:
            vector += w2vmodel[word].reshape( (1, num_of_features) )
            average_valence = np.mean(w2vmodel[word].reshape( (1, num_of_features) ))
            if max_valence < average_valence:
                max_valence = average_valence
            if min_valence > average_valence:
                min_valence = average_valence
            count += 1
        except KeyError:    # case where the token is not in the vocabulary(frequency of the token was < min_count)
            continue
    if count != 0:
        vector /= count
    if lex_1c != 0:
        lex_1 /= lex_1c
        vector = np.array([np.append(vector, np.array(lex_1))])
    else:
        vector = np.array([np.append(vector, np.array([[0]]))])
    if lex_2c != 0:
        lex_2 /= lex_2c
        vector = np.array([np.append(vector, np.array(lex_2))])
    else:
        vector = np.array([np.append(vector, np.array([[0]]))])
    average_valence = (max_valence + min_valence)/2
    vector = np.array([np.append(vector, np.array(average_valence))])
    return vector

def vader_sentiment_analysis(w2vmodel, original_tweets):
    i = 0
    model = []
    for vector in w2vmodel:
        if len(original_tweets[i]) > 0:
            model.append(np.append(vector, np.array(sentiment_analyzer_scores(original_tweets[i][0]))))
        else:
            model.append(np.append(vector, np.array(0)))
        i += 1
    model = np.array(model)
    return model