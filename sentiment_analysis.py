################################################################################
# Sentiment Analysis with Twitter Data - Comparing Machine Learning Algorithms #
################################################################################

# -*- coding: utf-8 -*-

import csv
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer

#function to process twitter data (for usage without professional sentiment lexicon --> with stemming)
def process(data):                                                      
    data_list = []                                                  
    punctuation =   ['.', ',', ';', '!', '?', '(', ')', '[', ']',                       #list of english punctuation marks (used in tweets)
                    '&', ':', '-', '/', '\\', '$', '*', '"', "'", '+',
                    '=', '@']                                      
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

#function to make a sentiment-lexicon based on twitter data
def make_lexicon(data, data_list):
    word_lexicon = []
    data_dict = {}   
    
    for tweet in data_list:                                                             #collect all different words
        for word in tweet:
            if word not in word_lexicon:
                word_lexicon.append(word)
    
    word_lexicon = sorted(word_lexicon)
    
    with open(data, 'r') as csvfile:                                                    #make dictionary with tweet as key and sentiment as value                     
        reader = csv.reader(csvfile, delimiter=';')
        for row in reader:
            for entry in data_list:
                key = ""
                for word in entry:
                    key = key + word + " "
                data_dict[key] = row[0]
        
    return word_lexicon
    
if __name__ == "__main__":
    data_list = process('data/comiccon_before_classified.csv')                          #!!! later: with all three data sets, merge them and make lexicon based on that
    print make_lexicon('data/comiccon_before_classified.csv', data_list)
