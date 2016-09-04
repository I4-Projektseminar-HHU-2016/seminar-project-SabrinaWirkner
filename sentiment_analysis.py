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

#global variables (P(class))
prob_class_pos = 0
prob_class_neut = 0
prob_class_neg = 0

#function to process twitter data (for usage without professional sentiment lexicon --> with stemming)
def process(data):           
    """
    We have three data sets (csvfiles) of twitter data containing the sentiment, date, tweet and hashtags.
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
                    '=', '@', '%', '~', '{', '}', '|']                                      
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
def make_sentiment_lexicon(data_list, data_dict):
    """
    To make a sentiment lexicon we need:
        - a lexicon of all words (word_lexicon)
        - the number of words in positive, neutral and negative tweets (words_in_pos_tweets, words_in_neut_tweets, words_in_neg_tweets)
        - the number of times a word occurs in a positive, neutral or negative tweet (pos, neut, neg)
        - the size of the word_lexicon
    
    With this we can estimate P(w|class) (without Laplace Smoothing) for all three classes. For the positive class this means:
        - w being pos
        - class being words_in pos_tweet
    
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
    
    number_tweets = pos_tweet_count + neut_tweet_count + neg_tweet_count                #number of tweets
    
    #estimate P(class)
    global prob_class_pos
    prob_class_pos = round((Decimal(pos_tweet_count) / number_tweets),5)
    global prob_class_neut 
    prob_class_neut = round((Decimal(neut_tweet_count) / number_tweets), 5)
    global prob_class_neg
    prob_class_neg = round((Decimal(neg_tweet_count) / number_tweets), 5)
    
    
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
        
        """
        #With Laplace-Smoothing - Will not be used as it caused problems (favorited uncommon class (negative))
        p_w_pos = ((pos + 1) / Decimal(words_in_pos_tweets + size_lexicon)) * 1000      
        p_w_neut = ((neut + 1) / Decimal(words_in_neut_tweets + size_lexicon)) * 1000
        p_w_neg = ((neg + 1) / Decimal(words_in_neg_tweets + size_lexicon)) * 1000
        """
        
        
        #Without Laplace-Smoothing
        if pos != 0:
            p_w_pos = pos / Decimal(words_in_pos_tweets) * 1000                         #P(w|class) will be multiplied by 1000 to have easier numbers to work with
        else:
            p_w_pos = 0                                                                 #for unseen class-word-occurence, the weight will be 1 (as 1 is neutral for multiplication)
        if neut != 0:
            p_w_neut = neut / Decimal(words_in_neut_tweets) * 1000                         
        else:
            p_w_neu = 0
        if neg != 0:
            p_w_neg = neg / Decimal(words_in_neg_tweets) * 1000                         
        else:
            p_w_neg = 0
        
        
        file.write(word + " " + str(round(p_w_pos,3)) + " " + str(round(p_w_neut, 3)) + " " + str(round(p_w_neg, 3)) + "\n")
    file.close()

#function to implement Naive Bayes algorithm
def do_naive_bayes(sentiment_lexicon, data_list, filename):
    """
    We use the Naive Bayes algorithm to determine the sentiment of each tweet.
    We work with our data_list containing all stemmed and processed tweets and our sentiment_lexicon.
    
    For Naive Bayes we need to:
        - estimate P(class|tweet) for every class (positive, neutral, negative) and every tweet
        - compare the probabilites to find greatest probability and therefore the most suitable sentiment
        - save the tweet with the respective sentiment in a csv-file (to later compare it with the manual sentiments)
    
    P(tweet|class) = P(w1|class) * P(w2|class) * ... * P(wn|class)
    
    Then we have to estimate P(class|tweet) by multiplying P(tweet|class) with P(class), which is
    determined by the number of positive, neutral and negative tweets in our data.
    """
    
    #results when using naive bayes algorithm will be saved as a csvfile
    with open(filename, 'wb') as new_file:
        writer = csv.writer(new_file, delimiter=';')
        for entry in data_list:
            p_tweet_pos = 1                                                                 #P(tweet|class)
            p_tweet_neut = 1
            p_tweet_neg = 1
            values = []
            sentiment = ""
            for word in entry:
                for line in open(sentiment_lexicon):
                    if word in line:
                        line_split = line.split()
                        if line_split[1] != "0.0":
                            p_tweet_pos *= Decimal(line_split[1])
                        else:
                            p_tweet_pos *= Decimal(0.025)                                   #multiply by a low weight to get best result with unseen word-class-occurences
                        if line_split[2] != "0.0":
                            p_tweet_neut *= Decimal(line_split[2])
                        else:
                            p_tweet_neut *= Decimal(0.025)
                        if line_split[3] != "0.0":
                            p_tweet_neg *= Decimal(line_split[3])
                        else:
                            p_tweet_neg *= Decimal(0.025)
            if p_tweet_pos != 1:
                pos_value = float(p_tweet_pos) * prob_class_pos                             #P(class|tweet)
            else:
                pos_value = 0
            if p_tweet_neut != 1:
                neut_value = float(p_tweet_neut) * prob_class_neut
            else:
                neut_value = 0
            if p_tweet_neg != 1:
                neg_value = float(p_tweet_neg) * prob_class_neg
            else:
                neg_value = 0
            values = [pos_value, neut_value, neg_value]
            if max(values) == values[0]:                                                    #determine sentiment (max P(class|tweet))
                sentiment = "positive"
            elif max(values) == values[1]:
                sentiment = "neutral"
            else:
                sentiment = "negative"
            tweet = ""
            for word in entry:
                tweet = tweet + word + " "
            writer.writerow([sentiment, tweet])                                             #write sentiment and tweet in csvfile
        
#function to implement Maximum Entropy Model algorithm
def do_max_ent(sentiment_lexicon, data_list, filename):
    """
    We use the Maximum Entropy Model algorithm to determine the sentiment of each tweet.
    We work with our data_list containing all stemmed and processed tweets and our sentiment_lexicon.
    
    For MEM we need to estimate the weighted feature sum for each class (positive, neutral, negative) and every tweet.
    For this we need to add up the weigths from our sentiment-lexicon.
    
    The weighted feature sums will be used in the MEM-algorithms. For example, for the positive class it's estimated like this:
        P(class|tweet) = e(pos_feature_sum) divided by (e(pos_feature_sum) + e(neut_feature_sum) + e(neg_feature_sum))
    """
    
    #results when using MEM algorithm will be saved as a csvfile
    with open(filename, 'wb') as new_file:
        writer = csv.writer(new_file, delimiter=';')
        for entry in data_list:
            pos_feature_sum = 0                                                                 #weighted feature sums
            neut_feature_sum = 0
            neg_feature_sum = 0
            sentiment = ""
            for word in entry:
                for line in open(sentiment_lexicon):
                    if word in line:
                        line_split = line.split()
                        pos_feature_sum += Decimal(line_split[1])
                        neut_feature_sum += Decimal(line_split[2])
                        neg_feature_sum += Decimal(line_split[3])
            #estimating P(class|tweet)
            p_pos_tweet = math.e**(float(pos_feature_sum)) / (math.e**(float(pos_feature_sum)) + math.e**(float(neut_feature_sum)) + math.e**(float(neg_feature_sum)))
            p_neut_tweet = math.e**(float(neut_feature_sum)) / (math.e**(float(pos_feature_sum)) + math.e**(float(neut_feature_sum)) + math.e**(float(neg_feature_sum)))
            p_neg_tweet = math.e**(float(neg_feature_sum)) / (math.e**(float(pos_feature_sum)) + math.e**(float(neut_feature_sum)) + math.e**(float(neg_feature_sum)))
            values = [p_pos_tweet, p_neut_tweet, p_neg_tweet]
            if max(values) == values[0]:                                                        #determine sentiment (max P(class|tweet))
                sentiment = "positive"
            elif max(values) == values[1]:
                sentiment = "neutral"
            else:
                sentiment = "negative"
            tweet = ""
            for word in entry:
                tweet = tweet + word + " "
            writer.writerow([sentiment, tweet])                                             #write sentiment and tweet in csvfile

#function to implement Support Vector Machina / k-nearest Neighbour algorithm
def do_svm(sentiment_lexicon, data_list, data_dict, filename):
    """
    We use the Support Vector Machine / k-nearest Neighbour algorithm to determine the sentiment of each tweet.
    We work with our data_list containing all stemmed and processed tweets and our sentiment_lexicon.
    
    First we need to estimate the vectors for each tweet. The values of the vector are the number of times each word of our lexicon appears in our tweet.
    Then we compare for each tweet its vector with all other vector to find the ones most similar. To do this, we need to estimate the normalized dot product.
    
    The sentiment of our tweet is estimated by the (intellectually assigned) sentiment of the k tweets with the most similar vector
    (k=3 or k=5, whichever gets the better results).
    """
    
    #create a list of lists (term_tweet_matrix) representing the vectors of each tweet
    term_tweet_matrix = []
    for tweet in data_list:
        vec_tweet = []
        for line in open(sentiment_lexicon):
            line_split = line.split()
            n = 0
            m = 0
            if line_split[0] in tweet:
                while m < len(tweet):
                    if line_split[0] == tweet[m]:
                        n += 1
                        m += 1
                    else:
                        m += 1
            vec_tweet.append(n)
        term_tweet_matrix.append(vec_tweet)
    
    #results when using SVM algorithm will be saved as a csvfile
    with open(filename, 'wb') as new_file:
        writer = csv.writer(new_file, delimiter=';')
        for init_vector in term_tweet_matrix:                                                       #estimate distances between all the vectors
            distances = []
            vec_len_first = 0
            for value in init_vector:
                vec_len_first = vec_len_first + value**(2)
            vec_len = math.sqrt(vec_len_first)                                                      #length of vector
            for vector in term_tweet_matrix:
                vec2_len_first = 0
                for value in vector:
                    vec2_len_first = vec2_len_first + value**(2)
                vec2_len = math.sqrt(vec2_len_first)                                                #length of vector we want to compare
                dot_product_first = 0
                i = 0
                while i < len(init_vector):
                    dot_product_first = dot_product_first + init_vector[i] * vector[i]
                    i += 1
                dot_product = dot_product_first / (vec_len * vec2_len)                              #normalized dot product
                distances.append(dot_product)                                                       #save all dot products into a list
            
            #find the 3 most similar vectors for our initial vector
            k1 = distances.index(max(distances))                                                    #k1 will be ignored here, but used later as it is the dot product with our inital vector (=1)
            distances[k1] = 0           
            k2 = distances.index(max(distances))
            distances[k2] = 0
            k3 = distances.index(max(distances))
            distances[k3] = 0
            k4 = distances.index(max(distances))
            tweet1 = ""
            tweet2 = ""
            tweet3 = ""
            for word in data_list[k2]:
                tweet1 = tweet1 + word + " "
            for word in data_list[k3]:
                tweet2 = tweet2 + word + " "
            for word in data_list[k4]:
                tweet3 = tweet3 + word + " "
            sentiments = []
            for key in data_dict:                                                                   #find the sentiments of the tweets corresponding to the most similar vectors
                if tweet1 == key:
                    sentiments.append(data_dict[key])
                elif tweet2 == key:
                    sentiments.append(data_dict[key])
                elif tweet3 == key:
                    sentiments.append(data_dict[key])
            pos = sentiments.count("positiv")
            neut = sentiments.count("neutral")
            neg = sentiments.count("negativ")
            sentiment = ""
            if pos == max([pos,neut,neg]):
                sentiment = "positiv"
            if neut == max([pos,neut,neg]):
                sentiment = "neutral"
            if neg == max([pos,neut,neg]):
                sentiment = "negativ"
            tweet = ""
            for word in data_list[k1]:
                tweet = tweet + word + " "
            writer.writerow([sentiment, tweet])

#function to implement Pointwise Mutual Information algorithm to estimate new weights
def do_pmi(sentiment_lexicon, data_list):
    """
    We use the Pointwise Mutual Information algorithm to determine new weights for our sentiment lexicon,
    which can then be used for our other methods (Naive Bayes, MEM, SVM).
    
    First we have to create a word-word-matrix of our lexicon, which displays how often words occur with each other.
    Usually a distance of around 4 words is used, but as we only have little data, we will count how often a word
    occurs with another word in a tweet.
    
    Based on this matrix we can estimate P(w) (word), P(c) (context-word) and P(w|c),
    which can be used for the PMI algorithm.
    
    We will use our original sentiment lexicon, than add to the weight of a word the weight of all other words multiplied by the estimated PMI between them.
    So if we have:
        great       1,5    0,8    0,02
        excellent   1,7    0,6    0,2
    And we have the PMI(great, excellent) = 0,6, we will do:
        great       1,5+0,6*1,7   0,8+0,6*0,6    + 0,02+0,6*0,2
    etc.

    The new sentiment lexicon will contain all words and their positive, neutral and negative weights:
        word    P(word|pos)    P(word|neut)    P(word|neg)
    """
    return
    
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
    make_sentiment_lexicon(data_list_joined, data_dict_joined)
    
    #do_naive_bayes('sentiment_lexicon.txt', data_list_before, 'sentiment_before_naive_bayes.csv')
    #do_max_ent('sentiment_lexicon.txt', data_list_before, 'sentiment_before_max_ent.csv')
    #do_svm('sentiment_lexicon.txt', data_list_before, data_dict_before, 'sentiment_before_svm.csv')
    do_pmi('sentiment_lexicon.txt', data_list_before)

        
        

