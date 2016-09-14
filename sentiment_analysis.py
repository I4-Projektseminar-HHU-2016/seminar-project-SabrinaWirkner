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
from tabulate import tabulate
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

#global variables (P(class))
prob_class_pos = 0
prob_class_neut = 0
prob_class_neg = 0
mac_precisions = {}
mic_precisions = {}
mac_recalls = {}
mic_recalls = {}
mac_accs = {}
mic_accs = {}

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
                sentiment = "positiv"
            elif max(values) == values[1]:
                sentiment = "neutral"
            else:
                sentiment = "negativ"
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
            e = Decimal(math.e)
            e_pos_feature_sum = e**(Decimal(pos_feature_sum))
            e_neut_feature_sum = e**(Decimal(neut_feature_sum))
            e_neg_feature_sum = e**(Decimal(neg_feature_sum))
            p_pos_tweet = Decimal(e_pos_feature_sum) / Decimal(e_pos_feature_sum + e_neut_feature_sum + e_neg_feature_sum)
            p_neut_tweet = Decimal(e_neut_feature_sum) / Decimal(e_pos_feature_sum + e_neut_feature_sum + e_neg_feature_sum)
            p_neg_tweet = Decimal(e_neg_feature_sum) / Decimal(e_pos_feature_sum + e_neut_feature_sum + e_neg_feature_sum)
            """
            p_pos_tweet = Decimal(math.e**(float(pos_feature_sum))) / Decimal((math.e**(float(pos_feature_sum))) + math.e**(float(neut_feature_sum)) + math.e**(float(neg_feature_sum)))
            p_neut_tweet = math.e**(float(neut_feature_sum)) / (math.e**(float(pos_feature_sum)) + math.e**(float(neut_feature_sum)) + math.e**(float(neg_feature_sum)))
            p_neg_tweet = math.e**(float(neg_feature_sum)) / (math.e**(float(pos_feature_sum)) + math.e**(float(neut_feature_sum)) + math.e**(float(neg_feature_sum)))
            """
            values = [p_pos_tweet, p_neut_tweet, p_neg_tweet]
            if max(values) == values[0]:                                                        #determine sentiment (max P(class|tweet))
                sentiment = "positiv"
            elif max(values) == values[1]:
                sentiment = "neutral"
            else:
                sentiment = "negativ"
            tweet = ""
            for word in entry:
                tweet = tweet + word + " "
            writer.writerow([sentiment, tweet])                                                 #write sentiment and tweet in csvfile

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
                if vec_len != 0 and vec2_len != 0:
                    dot_product = dot_product_first / (vec_len * vec2_len)                          #normalized dot product
                else:
                    dot_product = 0
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
            
#function to implement Pointwise Mutual Information algorithm to estimate new weights       REDO
def do_pmi(sentiment_lexicon, data_list):
    """
    We use the Pointwise Mutual Information algorithm to determine new weights for our sentiment lexicon,
    which can then be used for our other methods (Naive Bayes, MEM, SVM).
    
    First we have to create a word-word-matrix of our lexicon, which displays how often words occur with each other.
    Usually a distance of around 4 words is used, but as we only have little data, we will count how often a word
    occurs with another word in a tweet.
    
    The counts in the word-word-matrix will be replaced with joint probabilities. They are estimated as follows:
        count of contextword with word / counts of all contextwords with word
        multiplied with
        count of all contextwords with word / counts of all contextwords with all words
    For example you have the word "apple" and the contextword "pie". "apple" occurs 2 times with "pie" and 5 times in total with contextwords.
    All in all 32 occurences of words with contextwords are counted (only count either (apple | pie) or (pie | apple).
    So our algorithm for joint probability would be:
        (2 / 5) * (5 / 32)
    
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
    
    vocab_list = []                                                                         #list of vocabulary in lexicon
    for line in open(sentiment_lexicon):
        line_split = line.split()
        vocab_list.append(line_split[0])
    
    #creating the word_word_matrix, collecting all words the initial word occurs together with in a tweet
    word_word_matrix = []
    for voc in vocab_list:
        word_contextword = []
        for tweet in data_list:
            if voc in tweet:
                for word in tweet:
                    if word != voc:
                        word_contextword.append(word)
        word_contextword_dict = {}
        for element in word_contextword:
            word_contextword_dict[element] = word_contextword.count(element)
        word_contextword_list = []
        for vocab in vocab_list:
            if vocab in word_contextword_dict.keys():
                word_contextword_list.append(word_contextword_dict[vocab])
            else:
                word_contextword_list.append(0)
        word_word_matrix.append(word_contextword_list)
    
    #replace counts in word_word_matrix with joint probabilities
    all_counts = 0
    n = 0
    for element in word_word_matrix:
        for index, value in enumerate(element):
            if index >= n:
                all_counts += value
        n += 1
    joint_probability = 0
    for entry in word_word_matrix:
        all_contextword_counts = 0
        for value in entry:
            all_contextword_counts += value
        for i, element in enumerate(entry):
            if entry[i] != 0:
                joint_probability = (entry[i] / Decimal(all_contextword_counts)) * (all_contextword_counts / Decimal(all_counts))
            else:
                joint_probability = 0
            entry[i] = joint_probability

    #make a list for all words and their P(w) and P(c)
    probability_lists = []
    for i, entry in enumerate(word_word_matrix):
        voc = vocab_list[i]
        p_w = 0
        for value in entry:
            p_w += value
        p_c = 0
        for element in word_word_matrix:
            p_c += element[0]
        probability_list = [voc, p_w, p_c]
        probability_lists.append(probability_list)
    
    #make ppmi_matrix containing all ppmi-values for all words with all other words
    ppmi_matrix = []
    for i, entry in enumerate(word_word_matrix):
        ppmi_list = []
        for element in entry:
            word = probability_lists[i]
            p_w = word[1]
            p_c = word[2]
            if p_w != 0 and p_c != 0:
                x = element / Decimal((p_w * p_c))
                if x != 0:
                    pmi = math.log(x, 2)
                else:
                    pmi = 0
            else:
                pmi = 0
            ppmi = max(pmi, 0)
            ppmi_list.append(ppmi)
        ppmi_matrix.append(ppmi_list)

    file = open("sentiment_lexicon_pmi.txt", "w")
    word = ""
    p_w_pos = 0
    p_w_neut = 0
    p_w_neg = 0
    for i, line in enumerate(open(sentiment_lexicon)):
        line_split = line.split()
        word = line_split[0]
        p_w_pos = float(line_split[1])
        p_w_neut = float(line_split[2])
        p_w_neg = float(line_split[3])
        for index, element in enumerate(ppmi_matrix[i]):
            if element != 0:
                f = open(sentiment_lexicon)
                lines = f.readlines()
                cword = lines[index].split()
                pos_weight = float(cword[1])
                neut_weight = float(cword[2])
                neg_weight = float(cword[3])
                pos = element * pos_weight
                neut = element * neut_weight
                neg = element * neg_weight
                p_w_pos += pos
                p_w_neut += neut
                p_w_neg += neg
        file.write(word + " " + str(round(p_w_pos,3)) + " " + str(round(p_w_neut, 3)) + " " + str(round(p_w_neg, 3)) + "\n")
    file.close()

#function to estimate precision, recall and accuracy of our sentiment analysis
def analyze(int_data, data, name, filename):
    """
    To analyze the results of our sentiment analysis we will make use of the gold-system-labels matrix.
    (gold labels: sentiment assigned intellectually; system labels: sentiment assigned by algorithm)
    We will check how often our algorithm assigned a positive sentiment to a positive tweet, a neutral sentiment to a positive tweet etc.
    With these values we can determine the precision, recall and accuracy of our algorithm
    We will then make tables (confusion matrix, contingency tables, pooled table) of our data and save it as a txtfile.
    
    Confusion Matrix:                       Contingency Table (pos):    Pooled Table:
    NB      |  pos  |   neut   |  neg       pos | yes   | no                | yes   | no
    -----------------------------------     -------------------         --------------------
    pos     |       |          |            yes |       |               yes |       |   
    -----------------------------------     -------------------         --------------------
    neut    |       |          |            no  |       |               no  |       | 
    -----------------------------------
    neg     |       |          | 
    
    Macroaverage Precision (C.Tables):      Macroaverage Recall:        Macroaverage Accuracy:
    Microaverage Precision (P.Table):       Microaverage Recall:        Microaverage Accuracy:
    """
    
    with open(int_data, 'r') as csvfile:
        int_reader = csv.reader(csvfile, delimiter=';') 
        int_reader = list(int_reader) 
        pos_pos = 0                                                     #values for confusion matrix
        pos_neut = 0                                                    
        pos_neg = 0
        neut_pos = 0
        neut_neut = 0
        neut_neg = 0
        neg_pos = 0
        neg_neut = 0
        neg_neg = 0
        with open(data, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=';')  
            reader = list(reader)   
            for i, row in enumerate(int_reader):
                if row[0] == 'positiv':
                    if reader[i][0] == 'positiv':
                        pos_pos += 1
                    elif reader[i][0] == 'neutral':
                        pos_neut += 1
                    else:
                        pos_neg += 1
                elif row[0] == 'neutral':
                    if reader[i][0] == 'positiv':
                        neut_pos += 1
                    elif reader[i][0] == 'neutral':
                        neut_neut += 1
                    else:
                        neut_neg += 1
                else:
                    if reader[i][0] == 'positiv':
                        neg_pos += 1
                    elif reader[i][0] == 'neutral':
                        neg_neut += 1
                    else:
                        neg_neg += 1
    
    #values for contingency tables
    pos_y_y = pos_pos                                                                       #gold: positive, system: positive
    pos_y_n = pos_neut + pos_neg                                                            #gold: positive, system: neutral or negative
    pos_n_y = neut_pos + neg_pos                                                            #gold: neutral or negative, system: positive
    pos_n_n = neut_neut + neut_neg + neg_neut + neg_neg                                     #gold: neutral or negative, system: neutral or negative
    
    neut_y_y = neut_neut
    neut_y_n = neut_pos + neut_neg
    neut_n_y = pos_neut + neg_neut
    neut_n_n = pos_pos + pos_neg + neg_pos + neg_neg
    
    neg_y_y = neg_neg
    neg_y_n = neg_pos + neg_neut
    neg_n_y = pos_neg + neut_neg
    neg_n_n = pos_pos + pos_neut + neut_neut + neut_pos
    
    #values for pooled table
    y_y = pos_y_y + neut_y_y + neg_y_y
    y_n = pos_y_n + neut_y_n + neg_y_n
    n_y = pos_n_y + neut_n_y + neut_n_y
    n_n = pos_n_n + neut_n_n + neg_n_n
    
    pos_precision = pos_y_y / Decimal(pos_y_y + pos_n_y)
    neut_precision = neut_y_y / Decimal(neut_y_y + neut_n_y)
    neg_precision = neg_y_y / Decimal(neg_y_y + neg_n_y)
    mac_precision = (pos_precision + neut_precision + neg_precision) / 3                    #Macroaverage Precision
    mic_precision = y_y / Decimal(y_y + n_y)                                                #Microaverage Precision
    global mac_precisions
    mac_precisions[filename] = mac_precision
    global mic_precisions
    mic_precisions[filename] = mic_precision
    
    pos_recall = pos_y_y / Decimal(pos_y_y + pos_y_n)
    neut_recall = neut_y_y / Decimal(neut_y_y + neut_y_n)
    neg_recall = neg_y_y / Decimal(neg_y_y + neg_y_n)
    mac_recall = (pos_recall + neut_recall + neg_recall) / 3                                #Macroaverage Recall
    mic_recall = y_y / Decimal(y_y + y_n)                                                   #Microaverage Recall
    global mac_recalls
    mac_recalls[filename] = mac_recall
    global mic_recalls
    mic_recalls[filename] = mic_recall
    
    pos_acc = (pos_y_y + pos_n_n) / Decimal(pos_y_y + pos_y_n + pos_n_y + pos_n_n)
    neut_acc = (neut_y_y + neut_n_n) / Decimal(neut_y_y + neut_y_n + neut_n_y + neut_n_n)
    neg_acc = (neg_y_y + neg_n_n) / Decimal(neg_y_y + neg_y_n + neg_n_y + neg_n_n)
    mac_acc = (pos_acc + neut_acc + neg_acc) / 3                                            #Microaverage Accuracy
    mic_acc = (y_y + n_n) / Decimal(y_y + y_n + n_y + n_n)                                  #Macroaverage Accuracy
    global mac_accs
    mac_accs[filename] = mac_acc
    global mic_accs
    mic_accs[filename] = mic_acc
    
    #creating and saving the tables
    file = open(filename, "w")
    file.write(name + ': ' + '\n')
    
    file.write('Confusion Matrix:' + '\n')
    confusion_table = [['pos', pos_pos, neut_pos, neg_pos], ['neut', pos_neut, neut_neut, neg_neut], ['neg', pos_neg, neut_neg, neg_neg]]
    confusion_header = [name, 'pos', 'neut', 'neg']
    file.write(tabulate(confusion_table, confusion_header, tablefmt="grid") + '\n' + '\n')
    
    file.write('Contingency Table Positive:' + '\n')
    contingency_table_pos = [['yes', pos_y_y, pos_n_y], ['no', pos_y_n, pos_n_n]]
    contingency_header_pos = ['pos', 'yes', 'no']
    file.write(tabulate(contingency_table_pos, contingency_header_pos, tablefmt="grid") + '\n' + '\n')
    
    file.write('Contingency Table Neutral:' + '\n')
    contingency_table_neut = [['yes', neut_y_y, neut_n_y], ['no', neut_y_n, neut_n_n]]
    contingency_header_neut = ['neut', 'yes', 'no']
    file.write(tabulate(contingency_table_neut, contingency_header_neut, tablefmt="grid") + '\n' + '\n')
    
    file.write('Contingency Table Negative:' + '\n')
    contingency_table_neg = [['yes', neg_y_y, neg_n_y], ['no', neg_y_n, neg_n_n]]
    contingency_header_neg = ['neg', 'yes', 'no']
    file.write(tabulate(contingency_table_neg, contingency_header_neg, tablefmt="grid") + '\n' + '\n')
    
    file.write('Pooled Table:' + '\n')
    pooled_table = [['yes', y_y, n_y], ['no', y_n, n_n]]
    pooled_header = [' ', 'yes', 'no']
    file.write(tabulate(pooled_table, pooled_header, tablefmt="grid") + '\n' + '\n')
    
    file.write('Macroaverage Precision: ' + str(round(mac_precision, 3)) + '     ' + 'Macroaverage Recall: ' + str(round(mac_recall, 3)) + '     ' + 'Macroaverage Accuracy: ' + str(round(mac_acc, 3)) + '\n')
    file.write('Microaverage Precision: ' + str(round(mic_precision, 3)) + '     ' + 'Microaverage Recall: ' + str(round(mic_recall, 3)) + '     ' + 'Microaverage Accuracy: ' + str(round(mic_acc, 3)))
    file.close()

#function to visualize our results via plots
def visualize(int_data, data, data2, data3, name, name2, name3, filename):
    """
    To visualize our results we will create bar charts and pie charts, which will then be saved into one pdffile.
    """
    
    with PdfPages(filename) as pdf:
        #bar chart to compare the macro- and microaverage precision of Naive Bayes, MaxEnt and SVM
        num = 3
        mac_precisions_list = (mac_precisions['analysis_naive_bayes.txt'], mac_precisions['analysis_max_ent.txt'], mac_precisions['analysis_svm.txt'])
        x_locate = np.arange(num)
        width = 0.35 
        fig, ax = plt.subplots()
        rec = ax.bar(x_locate, mac_precisions_list, width, color='b')
        mic_precisions_list = (mic_precisions['analysis_naive_bayes.txt'], mic_precisions['analysis_max_ent.txt'], mic_precisions['analysis_svm.txt'])
        rec2 = ax.bar(x_locate + width, mic_precisions_list, width, color='g')
        ax.set_title('Macro- and Microaverage Precision of NB, MaxEnt and SVM')
        ax.set_xticks(x_locate + width)
        ax.set_xticklabels(('NB', 'MaxEnt', 'SVM'))
        ax.legend((rec[0], rec2[0]), ('Macroaverage Precision', 'Microaverage Precision'))
        axes = plt.gca()
        axes.set_ylim([0,1])
        pdf.savefig()
        plt.close()
        
        #bar chart to compare the macro- and microaverage recall of Naive Bayes, MaxEnt and SVM
        mac_recalls_list = (mac_recalls['analysis_naive_bayes.txt'], mac_recalls['analysis_max_ent.txt'], mac_recalls['analysis_svm.txt'])
        fig, ax = plt.subplots()
        rec = ax.bar(x_locate, mac_recalls_list, width, color='b')
        mic_recalls_list = (mic_recalls['analysis_naive_bayes.txt'], mic_recalls['analysis_max_ent.txt'], mic_recalls['analysis_svm.txt'])
        rec2 = ax.bar(x_locate + width, mic_recalls_list, width, color='g')
        ax.set_title('Macro- and Microaverage Recall of NB, MaxEnt and SVM')
        ax.set_xticks(x_locate + width)
        ax.set_xticklabels(('NB', 'MaxEnt', 'SVM'))
        ax.legend((rec[0], rec2[0]), ('Macroaverage Recall', 'Microaverage Recall'))
        axes = plt.gca()
        axes.set_ylim([0,1])
        pdf.savefig()
        plt.close()
        
        #bar chart to compare the macro- and microaverage accuracy of Naive Bayes, MaxEnt and SVM
        mac_accs_list = (mac_accs['analysis_naive_bayes.txt'], mac_accs['analysis_max_ent.txt'], mac_accs['analysis_svm.txt'])
        fig, ax = plt.subplots()
        rec = ax.bar(x_locate, mac_accs_list, width, color='b')
        mic_accs_list = (mic_accs['analysis_naive_bayes.txt'], mic_accs['analysis_max_ent.txt'], mic_accs['analysis_svm.txt'])
        rec2 = ax.bar(x_locate + width, mic_accs_list, width, color='g')
        ax.set_title('Macro- and Microaverage Accuracy of NB, MaxEnt and SVM')
        ax.set_xticks(x_locate + width)
        ax.set_xticklabels(('NB', 'MaxEnt', 'SVM'))
        ax.legend((rec[0], rec2[0]), ('Macroaverage Accuracy', 'Microaverage Accuracy'))
        axes = plt.gca()
        axes.set_ylim([0,1])
        pdf.savefig()
        plt.close()
        
        #pie chart to show the proportion between positive, neutral and negative sentiment assigned intellectually
        pos = 0
        neut = 0
        neg = 0
        with open(int_data, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=';')       
            for row in reader:
                if row[0] == 'positiv':
                    pos += 1
                elif row[0] == 'neutral':
                    neut += 1
                else:
                    neg += 1
        p_pos = (pos / Decimal(pos + neut + neg)) * 100
        p_neut = (neut / Decimal(pos + neut + neg)) * 100
        p_neg = (neg / Decimal(pos + neut + neg)) * 100
        labels = ['positive', 'neutral', 'negative']
        values = [p_pos, p_neut, p_neg]
        colors = ['yellowgreen', 'gold', 'lightskyblue']
        plt.pie(values, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=90)
        plt.axis('equal')
        plt.title('Intellectually Assigned Sentiment', y=1.05)
        pdf.savefig()
        plt.close()
        
        #pie chart to show the proportion between positive, neutral and negative sentiment assigned by Naive Bayes
        pos = 0
        neut = 0
        neg = 0
        with open(data, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=';')       
            for row in reader:
                if row[0] == 'positiv':
                    pos += 1
                elif row[0] == 'neutral':
                    neut += 1
                else:
                    neg += 1
        p_pos = (pos / Decimal(pos + neut + neg)) * 100
        p_neut = (neut / Decimal(pos + neut + neg)) * 100
        p_neg = (neg / Decimal(pos + neut + neg)) * 100
        labels = ['positive', 'neutral', 'negative']
        values = [p_pos, p_neut, p_neg]
        colors = ['yellowgreen', 'gold', 'lightskyblue']
        plt.pie(values, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=90)
        plt.axis('equal')
        title = 'Sentiment Assigned by ' + name
        plt.title(title, y=1.05)
        pdf.savefig()
        plt.close()
        
        #pie chart to show the proportion between positive, neutral and negative sentiment assigned by MaxEnt
        pos = 0
        neut = 0
        neg = 0
        with open(data2, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=';')       
            for row in reader:
                if row[0] == 'positiv':
                    pos += 1
                elif row[0] == 'neutral':
                    neut += 1
                else:
                    neg += 1
        p_pos = (pos / Decimal(pos + neut + neg)) * 100
        p_neut = (neut / Decimal(pos + neut + neg)) * 100
        p_neg = (neg / Decimal(pos + neut + neg)) * 100
        labels = ['positive', 'neutral', 'negative']
        values = [p_pos, p_neut, p_neg]
        colors = ['yellowgreen', 'gold', 'lightskyblue']
        plt.pie(values, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=90)
        plt.axis('equal')
        title = 'Sentiment Assigned by ' + name2
        plt.title(title, y=1.05)
        pdf.savefig()
        plt.close()
        
        #pie chart to show the proportion between positive, neutral and negative sentiment assigned by SVM
        pos = 0
        neut = 0
        neg = 0
        with open(data3, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=';')       
            for row in reader:
                if row[0] == 'positiv':
                    pos += 1
                elif row[0] == 'neutral':
                    neut += 1
                else:
                    neg += 1
        p_pos = (pos / Decimal(pos + neut + neg)) * 100
        p_neut = (neut / Decimal(pos + neut + neg)) * 100
        p_neg = (neg / Decimal(pos + neut + neg)) * 100
        labels = ['positive', 'neutral', 'negative']
        values = [p_pos, p_neut, p_neg]
        colors = ['yellowgreen', 'gold', 'lightskyblue']
        plt.pie(values, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=90)
        plt.axis('equal')
        title = 'Sentiment Assigned by ' + name3
        plt.title(title, y=1.05)
        pdf.savefig()
        plt.close()
        
        #bar chart to compare the macroaverage results of the algorithms with and without PMI
        num = 4
        precisions_list = (mac_precisions['analysis_naive_bayes.txt'], mac_precisions['analysis_naive_bayes_pmi.txt'], mac_precisions['analysis_max_ent.txt'], 
                            mac_precisions['analysis_max_ent_pmi.txt'])
        x_locate = np.arange(num)
        width = 0.3 
        fig, ax = plt.subplots()
        rec = ax.bar(x_locate, precisions_list, width, color='b')
        recalls_list = (mac_recalls['analysis_naive_bayes.txt'], mac_recalls['analysis_naive_bayes_pmi.txt'], mac_recalls['analysis_max_ent.txt'], 
                        mac_recalls['analysis_max_ent_pmi.txt'])
        rec2 = ax.bar(x_locate + width, recalls_list, width, color='g')
        accuracy_list = (mac_accs['analysis_naive_bayes.txt'], mac_accs['analysis_naive_bayes_pmi.txt'], mac_accs['analysis_max_ent.txt'], 
                        mac_accs['analysis_max_ent_pmi.txt'])
        rec3 = ax.bar(x_locate + width + width, accuracy_list, width, color='r')
        ax.set_title('Macroaverage Precision, Recall and Accuracy with and without PMI')
        ax.set_xticks(x_locate + width)
        ax.set_xticklabels(('NB', 'NB PMI', 'MaxEnt', 'MaxEnt PMI'))
        ax.legend((rec[0], rec2[0], rec3[0]), ('Precision', 'Recall', 'Accuracy'))
        axes = plt.gca()
        axes.set_ylim([0,1])
        pdf.savefig()
        plt.close()
        
        #bar chart to compare the microaverage results of the algorithms with and without PMI
        num = 4
        precisions_list = (mic_precisions['analysis_naive_bayes.txt'], mic_precisions['analysis_naive_bayes_pmi.txt'], mic_precisions['analysis_max_ent.txt'], 
                            mic_precisions['analysis_max_ent_pmi.txt'])
        x_locate = np.arange(num)
        width = 0.3 
        fig, ax = plt.subplots()
        rec = ax.bar(x_locate, precisions_list, width, color='b')
        recalls_list = (mic_recalls['analysis_naive_bayes.txt'], mic_recalls['analysis_naive_bayes_pmi.txt'], mic_recalls['analysis_max_ent.txt'], 
                        mic_recalls['analysis_max_ent_pmi.txt'])
        rec2 = ax.bar(x_locate + width, recalls_list, width, color='g')
        accuracy_list = (mic_accs['analysis_naive_bayes.txt'], mic_accs['analysis_naive_bayes_pmi.txt'], mic_accs['analysis_max_ent.txt'], 
                        mic_accs['analysis_max_ent_pmi.txt'])
        rec3 = ax.bar(x_locate + width + width, accuracy_list, width, color='r')
        ax.set_title('Microaverage Precision, Recall and Accuracy with and without PMI')
        ax.set_xticks(x_locate + width)
        ax.set_xticklabels(('NB', 'NB PMI', 'MaxEnt', 'MaxEnt PMI'))
        ax.legend((rec[0], rec2[0], rec3[0]), ('Precision', 'Recall', 'Accuracy'))
        axes = plt.gca()
        axes.set_ylim([0,1])
        pdf.savefig()
        plt.close()

if __name__ == "__main__":
    #Only the first set of twitter data will be used to ensure the shortest time for processing!
    #process data of all three data sets
    data_list_before = process('data/comiccon_before_classified.csv')
    data_list_during = process('data/comiccon_during_classified.csv')
    data_list_after = process('data/comiccon_after_classified.csv')
    print "Twitter Data processed. Creating Dictionaries."

    #make dictionaries with sentiment of all three data sets
    data_dict_before = make_dictionary('data/comiccon_before_classified.csv', data_list_before)
    data_dict_during = make_dictionary('data/comiccon_during_classified.csv', data_list_during)
    data_dict_after = make_dictionary('data/comiccon_after_classified.csv', data_list_after)
    print "Dictionaries created. Merging."

    #merge data_lists and data_dicts to make sentiment lexicon
    data_list_joined = data_list_before + data_list_during + data_list_after               
    data_dict_joined = dict(data_dict_before.items() + data_dict_during.items() + data_dict_after.items())
    print "Done merging. Creating Sentiment Lexicon."
    
    #create sentiment lexicon (txtfile) based on twitter data
    make_sentiment_lexicon(data_list_joined, data_dict_joined)
    print "Created Sentiment Lexicon. Creating 2nd Sentiment Lexicon using PMI. This will take approximately 20 minutes."

    #create a second sentiment lexicon using PMI
    do_pmi('sentiment_lexicon.txt', data_list_joined)
    print "Created 2nd Sentiment Lexicon. Using Naive Bayes Algorithm."
    
    #the results using the algorithms will be saved as a csvfile
    #use the naive bayes algorithm with and without PMI
    do_naive_bayes('sentiment_lexicon.txt', data_list_before, 'sentiment_before_naive_bayes.csv')
    do_naive_bayes('sentiment_lexicon_pmi.txt', data_list_before, 'sentiment_before_naive_bayes_pmi.csv')
    print "Finished using Naive Bayes. Using Maximum Entropy Algorithm."
    
    #use the max ent algorithm with and without PMI
    do_max_ent('sentiment_lexicon.txt', data_list_before, 'sentiment_before_max_ent.csv')
    do_max_ent('sentiment_lexicon_pmi.txt', data_list_before, 'sentiment_before_max_ent_pmi.csv')
    print "Finished using MaxEnt. Using SVM Algorithm. This will take several minutes."
    
    #use the svm algorithm
    do_svm('sentiment_lexicon.txt', data_list_before, data_dict_before, 'sentiment_before_svm.csv')
    print "Finished using SVM. Analyzing Data."
    
    #analyze the results and save the analysis as a txtfile
    analyze('data/comiccon_before_classified.csv', 'sentiment_before_naive_bayes.csv', 'Naive Bayes', 'analysis_naive_bayes.txt')
    analyze('data/comiccon_before_classified.csv', 'sentiment_before_naive_bayes_pmi.csv', 'Naive Bayes PMI', 'analysis_naive_bayes_pmi.txt')
    analyze('data/comiccon_before_classified.csv', 'sentiment_before_max_ent.csv', 'Max Ent', 'analysis_max_ent.txt')
    analyze('data/comiccon_before_classified.csv', 'sentiment_before_max_ent_pmi.csv', 'Max Ent PMI', 'analysis_max_ent_pmi.txt')
    analyze('data/comiccon_before_classified.csv', 'sentiment_before_svm.csv', 'SVM', 'analysis_svm.txt')
    print "Done analyzing. Visualizing Analysis."
    
    #visualize the analysis and save it as a pdffile
    visualization = 'visualization.pdf'
    visualize('data/comiccon_before_classified.csv', 'sentiment_before_naive_bayes.csv', 'sentiment_before_max_ent.csv', 'sentiment_before_svm.csv', 
                'Naive Bayes', 'Maximum Entropy', 'Support Vector Machine', visualization)
    print "Done."

    """
    ##########################################
    # Long Version using the Joined Data Set #
    ##########################################
    
    #process data of all three data sets
    data_list_before = process('data/comiccon_before_classified.csv')
    data_list_during = process('data/comiccon_during_classified.csv')
    data_list_after = process('data/comiccon_after_classified.csv')
    print "Twitter Data processed. Creating Dictionaries."

    #make dictionaries with sentiment of all three data sets
    data_dict_before = make_dictionary('data/comiccon_before_classified.csv', data_list_before)
    data_dict_during = make_dictionary('data/comiccon_during_classified.csv', data_list_during)
    data_dict_after = make_dictionary('data/comiccon_after_classified.csv', data_list_after)
    print "Dictionaries created. Merging."

    #merge data_lists and data_dicts to make sentiment lexicon
    data_list_joined = data_list_before + data_list_during + data_list_after               
    data_dict_joined = dict(data_dict_before.items() + data_dict_during.items() + data_dict_after.items())
    print "Done merging. Creating Sentiment Lexicon."
    
    #create sentiment lexicon (txtfile) based on twitter data
    make_sentiment_lexicon(data_list_joined, data_dict_joined)
    print "Created Sentiment Lexicon. Creating 2nd Sentiment Lexicon using PMI. This will take approximately 20 minutes."

    #create a second sentiment lexicon using PMI
    do_pmi('sentiment_lexicon.txt', data_list_joined)
    print "Created 2nd Sentiment Lexicon. Using Naive Bayes Algorithm. This will take a couple of minutes."
    
    #the results using the algorithms will be saved as a csvfile
    #use the naive bayes algorithm with and without PMI
    do_naive_bayes('sentiment_lexicon.txt', data_list_joined, 'sentiment_joined_naive_bayes.csv')
    do_naive_bayes('sentiment_lexicon_pmi.txt', data_list_joined, 'sentiment_joined_naive_bayes_pmi.csv')
    print "Finished using Naive Bayes. Using Maximum Entropy Algorithm. This will take a couple of minutes."
    
    #use the max ent algorithm with and without PMI
    do_max_ent('sentiment_lexicon.txt', data_list_joined, 'sentiment_joined_max_ent.csv')
    do_max_ent('sentiment_lexicon_pmi.txt', data_list_before, 'sentiment_before_max_ent_pmi.csv')
    print "Finished using MaxEnt. Using SVM Algorithm. This will take approximately an hour."
    
    #use the svm algorithm
    do_svm('sentiment_lexicon.txt', data_list_joined, data_dict_joined, 'sentiment_joined_svm.csv')
    print "Finished using SVM. Analyzing Data."
    
    analyze('data/comiccon_joined_classified.csv', 'sentiment_joined_naive_bayes.csv', 'Naive Bayes', 'analysis_joined_naive_bayes.txt')
    analyze('data/comiccon_joined_classified.csv', 'sentiment_joined_naive_bayes_pmi.csv', 'Naive Bayes PMI', 'analysis_joined_naive_bayes_pmi.txt')
    analyze('data/comiccon_joined_classified.csv', 'sentiment_joined_max_ent.csv', 'Max Ent', 'analysis_joined_max_ent.txt')
    analyze('data/comiccon_joined_classified.csv', 'sentiment_joined_max_ent_pmi.csv', 'Max Ent PMI', 'analysis_joined_max_ent_pmi.txt')
    analyze('data/comiccon_joined_classified.csv', 'sentiment_joined_svm.csv', 'SVM', 'analysis_joined_svm.txt')
    print "Done analyzing. Visualizing Analysis."
    
    #visualize the analysis and save it as a pdffile
    visualization = 'visualization_joined.pdf'
    visualize('data/comiccon_joined_classified.csv', 'sentiment_joined_naive_bayes.csv', 'sentiment_joined_max_ent.csv', 'sentiment_joined_svm.csv', 
                'Naive Bayes', 'Maximum Entropy', 'Support Vector Machine', visualization)
    print "Done."
    """
    
    """
    #######################################################
    # Analyzing the After and During sets of Twitter Data #
    #######################################################
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

    #create a second sentiment lexicon using PMI
    do_pmi('sentiment_lexicon.txt', data_list_joined)
    
    #the results using the algorithms will be saved as a csvfile
    do_naive_bayes('sentiment_lexicon.txt', data_list_during, 'sentiment_during_naive_bayes.csv')
    do_naive_bayes('sentiment_lexicon_pmi.txt', data_list_during, 'sentiment_during_naive_bayes_pmi.csv')
    do_naive_bayes('sentiment_lexicon.txt', data_list_after, 'sentiment_after_naive_bayes.csv')
    do_naive_bayes('sentiment_lexicon_pmi.txt', data_list_after, 'sentiment_after_naive_bayes_pmi.csv')
    do_max_ent('sentiment_lexicon.txt', data_list_during, 'sentiment_during_max_ent.csv')
    do_max_ent('sentiment_lexicon_pmi.txt', data_list_during, 'sentiment_during_max_ent_pmi.csv')
    do_max_ent('sentiment_lexicon.txt', data_list_after, 'sentiment_after_max_ent.csv')
    do_max_ent('sentiment_lexicon_pmi.txt', data_list_after, 'sentiment_after_max_ent_pmi.csv')
    do_svm('sentiment_lexicon.txt', data_list_during, data_dict_during, 'sentiment_during_svm.csv')
    do_svm('sentiment_lexicon.txt', data_list_after, data_dict_during, 'sentiment_after_svm.csv')

    #analyze the results
    analyze('data/comiccon_during_classified.csv', 'sentiment_during_naive_bayes.csv', 'Naive Bayes', 'analysis_during_naive_bayes.txt')
    analyze('data/comiccon_during_classified.csv', 'sentiment_during_naive_bayes_pmi.csv', 'Naive Bayes PMI', 'analysis_during_naive_bayes_pmi.txt')
    analyze('data/comiccon_during_classified.csv', 'sentiment_during_max_ent.csv', 'Max Ent', 'analysis_during_max_ent.txt')
    analyze('data/comiccon_during_classified.csv', 'sentiment_during_max_ent_pmi.csv', 'Max Ent PMI', 'analysis_during_max_ent_pmi.txt')
    analyze('data/comiccon_during_classified.csv', 'sentiment_during_svm.csv', 'SVM', 'analysis_during_svm.txt')
    analyze('data/comiccon_after_classified.csv', 'sentiment_after_naive_bayes.csv', 'Naive Bayes', 'analysis_after_naive_bayes.txt')
    analyze('data/comiccon_after_classified.csv', 'sentiment_after_naive_bayes_pmi.csv', 'Naive Bayes PMI', 'analysis_after_naive_bayes_pmi.txt')
    analyze('data/comiccon_after_classified.csv', 'sentiment_after_max_ent.csv', 'Max Ent', 'analysis_during_max_ent.txt')
    analyze('data/comiccon_after_classified.csv', 'sentiment_after_max_ent_pmi.csv', 'Max Ent PMI', 'analysis_after_max_ent_pmi.txt')
    analyze('data/comiccon_after_classified.csv', 'sentiment_after_svm.csv', 'SVM', 'analysis_after_svm.txt')
    
    #visualize the analysis and save it as a pdffile
    visualization = 'visualization_during.pdf'
    visualization2 = 'visualization_after.pdf
    visualize('data/comiccon_during_classified.csv', 'sentiment_during_naive_bayes.csv', 'sentiment_during_max_ent.csv', 'sentiment_during_svm.csv', 
                'Naive Bayes', 'Maximum Entropy', 'Support Vector Machine', visualization)
    visualize('data/comiccon_after_classified.csv', 'sentiment_after_naive_bayes.csv', 'sentiment_after_max_ent.csv', 'sentiment_after_svm.csv', 
                'Naive Bayes', 'Maximum Entropy', 'Support Vector Machine', visualization2)
    """

