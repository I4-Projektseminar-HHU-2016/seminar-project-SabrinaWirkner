################################################################################
# Sentiment Analysis with Twitter Data - Comparing Machine Learning Algorithms #
################################################################################

# -*- coding: utf-8 -*-

import csv
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
import math
from decimal import *

#function to process twitter data (for usage without professional sentiment lexicon --> with stemming)
def process(data):           
    """
    Before we can analyize our data, we need to process it.
    This includes:
        - converting all words to lowercase
        - deleting punctuation marks (emoticons can contain sentiment, but this is a problem that may be adressed later)
        - deleting numbers
        - tokenize tweets to get a list of words
        - deleting stopwords
        - stem all words
    
    This function will return a list of lists containing all processed tweets (data_list).
    """                                           
    data_list = []                                                  
    punctuation =   ['.', ',', ';', '!', '?', '(', ')', '[', ']',                       #list of english punctuation marks (used in tweets)
                    '&', ':', '-', '/', '\\', '$', '*', '"', "'", '+',
                    '=', '@', '%', '~']                                      
    stopwords = nltk.corpus.stopwords.words("english")                                  #list of stopwords
    
    with open(data, 'r') as csvfile:                                                    #collect tweets from csv-data in list
        reader = csv.reader(csvfile, delimiter=';')       
        for row in reader:
            data_list.append(row[2])
            
    for index, element in enumerate(data_list):
        element = element.lower()                                                       #tweet to lowercase           
        for mark in punctuation:
            element = element.replace(mark, '')                                         #delete punctuation marks
        element = ''.join([i for i in element if not i.isdigit()])                      #delete numbers
        element = word_tokenize(element)                                                #tokenize tweet
        element = [w for w in element if w not in stopwords]                            #delete stopwords
        
        for i, word in enumerate(element):                                              #stem words in tweet
            word = SnowballStemmer("english").stem(word)
            element[i] = str(word)
            
        data_list[index] = element
    
    """
    with open(data, 'r') as csvfile:                                                    #tried to write processed tweets in csvfile
        reader = csv.reader(csvfile, delimiter=';')
        for row in reader:
            new_row = []
            for el in row:
                for word in element:
                    new_row.append(word)
                    new_row.append(" ")
        
            with open('data/processed_comiccon_before_classified.csv', 'a') as csvfile:
                writer = csv.writer(csvfile, delimiter=';')
                writer.writerow(new_row)"""
        
    return data_list

#function to process twitter data (for usage with professional sentiment lexicon --> without stemming)
def process_for_lexicon(data):  
    """
    If we don't make our own sentiment lexicon, but work with a professional one, we still need to process our data,
    but we don't have to stem the words (as the sentiment lexicon contains all flexed word form etc.).
    
    This function will return a list of lists containing all processed tweets (data_list).
    """                                        
    data_list = []                                                                      #collect tweets from csv-data in list
    punctuation =   ['.', ',', ';', '!', '?', '(', ')', '[', ']',
                    '&', ':', '-', '/', '\\']                                           #list of english punctuation marks (used in tweets)
    stopwords = nltk.corpus.stopwords.words("english")                                  #list of stopwords
    
    with open(data, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')       
        for row in reader:
            data_list.append(row[2])
    
    for index, element in enumerate(data_list):
        element = element.lower()                                                       #tweet to lowercase           
        for mark in punctuation:
            element = element.replace(mark, '')                                         #delete punctuation marks
        element = ''.join([i for i in element if not i.isdigit()])                      #delete numbers
        element = word_tokenize(element)                                                #tokenize tweet
        element = [w for w in element if w not in stopwords]                            #delete stopwords

        data_list[index] = element
            
    return data_list

#function to make dictionaries with data and their sentiment
def make_dictionary(data, data_list):
    data_dict = {}
    
    #make dictionary with tweet as key and sentiment as value      
    with open(data, 'r') as csvfile:                                                                   
        reader = csv.reader(csvfile, delimiter=';')
        n = 0
        for row in reader:
            key = ""
            for word in data_list[n]:
                key = key + word + " "
            data_dict[key] = row[0]
            n +=1
    
    return data_dict
    
#function to make a sentiment-lexicon based on twitter data
def make_sentiment_lexicon(data, data_list, data_dict):
    """
    To make a sentiment lexicon we need:
        - a lexicon of all words (word_lexicon)
        - the number of words in positive, neutral and negative tweets (words_in_pos_tweets, words_in_neut_tweets, words_in_neg_tweets)
        - the number of times a word occurs in a positive, neutral or negative tweet (pos, neut, neg)
        - the size of the word_lexicon
    
    With this we can estimate P(w|class) with Laplace Smoothing for all three classes. For the positive class this means:
        - w being pos + 1
        - class being words_in pos_tweet + size of word_lexicon
    
    The results will be written in a txtfile (our sentiment lexicon), looking like this:
        word    P(word|pos)    P(word|neut)    P(word|neg)
    """
    word_lexicon = []
    
    #collect all different words
    for tweet in data_list:                                                            
        for word in tweet:
            if word not in word_lexicon:
                word_lexicon.append(word)
    
    word_lexicon = sorted(word_lexicon)                                                 #lexicon containing all words in alphabetical order
    
    #various information about data
    size_lexicon = len(word_lexicon)                                                    #number of words in lexicon
    pos_tweet_count = 0                                                                 #number of positive tweets, number of words in positive tweets
    words_in_pos_tweets = 0
    neut_tweet_count = 0                                                                #number of neutral tweets. number of words in neutral tweets
    words_in_neut_tweets = 0
    neg_tweet_count = 0                                                                 #number of negative tweets, number of words in negative tweets
    words_in_neg_tweets = 0
    for key in data_dict:
        if data_dict[key] == "positiv":
            pos_tweet_count += 1
            for word in key:
                words_in_pos_tweets +=1
        elif data_dict[key] == "neutral":
            neut_tweet_count += 1
            for word in key:
                words_in_neut_tweets +=1
        else:
            neg_tweet_count += 1
            for word in key:
                words_in_neg_tweets +=1
    
    #make sentiment-lexicon
    file = open("sentiment_lexicon.txt", "w")
    for word in word_lexicon:
        pos = 0
        neut = 0
        neg = 0
        for key in data_dict:                                                           #number of times word occurs in positive, neutral and negative tweets
            if word in key:
                if data_dict[key] == "positiv":
                    pos += 1
                elif data_dict[key] == "neutral":
                    neut += 1
                else:
                    neg += 1
        
        p_w_pos = ((pos + 1) / Decimal(words_in_pos_tweets + size_lexicon)) * 1000      #p(w|class) will be multiplied by 1000 to have easier numbers to work with
        p_w_neut = ((neut + 1) / Decimal(words_in_neut_tweets + size_lexicon)) * 1000
        p_w_neg = ((neg + 1) / Decimal(words_in_neg_tweets + size_lexicon)) * 1000
        
        #print round(p_w_pos,5), round(p_w_neut), round(p_w_neg)
        
        file.write(word + " " + str(round(p_w_pos,5)) + " " + str(round(p_w_neut, 5)) + " " + str(round(p_w_neg, 5)) + "\n")
    file.close()
    
if __name__ == "__main__":
    #process data of all three data sets
    data_list_before = process('data/comiccon_before_classified.csv')
    data_list_during = process('data/comiccon_during_classified.csv')
    data_list_after = process('data/comiccon_after_classified.csv')
    
    #make dictionaries with sentiment of all three data sets
    data_dict_before = make_dictionary('data/comiccon_before_classified.csv', data_list_before)
    data_dict_during = make_dictionary('data/comiccon_during_classified.csv', data_list_during)
    data_dict_after = make_dictionary('data/comiccon_after_classified.csv', data_list_after)
    
    #merge data_lists and data_dicts to make sentiment lexicon
    data_list_joined = data_list_before + data_list_during + data_list_after               
    data_dict_joined = dict(data_dict_before.items() + data_dict_during.items() + data_dict_after.items())
    
    #create sentiment lexicon (txtfile) based on twitter data
    make_sentiment_lexicon('data/comiccon_before_classified.csv', data_list_joined, data_dict_joined)
