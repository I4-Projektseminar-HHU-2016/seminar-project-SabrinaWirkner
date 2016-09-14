# Project Title

Sentiment Analysis with Twitter Data - Comparison of Machine Learning Algorithms

A project to assign a sentiment (positive, neutral, negative) to tweets and compare the results with intellectual classification.
Two sentiment lexica will be created, one with using Pointwise Mutual Information and one without.
To determine the sentiment the following Machine Learning Algorithms will be used and their results compares:
- [x] Naive Bayes
- [x] Maximum Entropy Models
- [x] Support Vector Machine

The results will be saved as CSVfiles. The analysis of the results of each algorithm will be saved as a TXTfile and the visualization with charts as one PDFfile.


## Features ##

- [x] Language Processing of Twitter Data
- [x] Creating a Sentiment Lexicon with weights for positive, neutral an negative sentiment based on intellectually classified Twitter Data
- [x] Creating a Sentiment Lexicon wiht weights based on 1st Lexicon, but with Pointwise Mutual Information
- [x] Implemented Naive Bayes Algorithm, modified to suit our data and needs
- [x] Implemented Maximum Entropy Algorithm, modified to suit our data and needs
- [x] Implemented Support Vector Machine Algorithm, modified to suit our data and needs
- [x] Saving results as CSVfiles containing the tweets and their assigned sentiment
- [x] Analyzing the results by creating a Confusion Matrix, Contingency Tables and a Pooled Table and estimating macro- and microaverage Precision, Recall and Accuracy
- [x] Saving analyses containing the tables and values as TXTfiles
- [x] Visualizing the analysis with bar charts and pie charts, comparing the algorithms, the distribution of sentiments and the usage of PMI
- [x] Saving the visualization as a PDFfile

Unfinished / Future Features:
- [ ] Comparison between the three sets of Twitter Data
- [ ] Comparison with using professional Sentiment Lexica
- [ ] Laplace Smoothing
- [ ] GUI to select own data, what algorithms to use & to compare and to directily show analysis and visualization
- [ ] Improve algorithms by solving problems such as negation, meaning of punctuation, irony, emoticons, abbreviations and colloquial language

## Getting Started

Clone / Download the repository.
It's important you have the program (sentiment_analysis.py) and the Twitter Data (data).
If you only wish to have the results, but not run the program yourselve, download the results folder, which contains all CSV-, TXT- and PDFfiles.

### Prerequisities

Install tabulate in your console:

```
pip install tabulate
```

Make sure everything is imported correctly. 
If there are problems, you might also need to install nltk, csv, matplotlib, numpy or math if you've never used them.

### Installing

```
Start the program.
```

The program will use the algorithms to analyze the smallest of the three sets of Twitter Data (~400 tweets).
The console will show what is done and what the program is doing at the moment. You will also be notified if a specific step will take some time.
All in all the program will run for about 30 minutes. When it is finished, the console will show "Done.".
After running the program, you will find in your folder CSVfiles containing the assigned sentiments using Naive Bayes and Co., TXTfiles with the tables and values of the analysis and a PDFfile showing all bar charts and pie charts.

If you wish to analyze all Twitter Data (~2000 Tweets):

```
1. Comment out the code in the main function
2. Delete the commentation of the second block ("Long Version using the Joined Data Set")
3. Start the program
```

Everything will be done the same way, but the program will run for over an hour, so it is not the first choice.

If you wish to also analyze the other two data sets (~700 Tweets and ~800 Tweets) seperately:

```
1. Comment out the first two blocks of code in the main function
2. Delete the commentation of the last block ("Analyzing the After and During Sets of Twitter Data")
3. Start the program
```

Everything will be done the same way, but the program will run for quite a while, so it is not the first choice.

All three parts of the main function can be run in succession. The files will not be overwritten, as new ones are created.

## Versioning

Sentiment Analysis 1.0.0

## Authors

* **Sabrina Wirkner** - [HHU-Wirkner](https://github.com/SabrinaWirkner)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

* Hat tip to anyone who's code was used
* Inspiration
* etc

*template inspired by [https://gist.github.com/PurpleBooth/109311bb0361f32d87a2](https://gist.github.com/PurpleBooth/109311bb0361f32d87a2)*
