**Welcome to my repository**. This is a twitter feed analyser for making multi-class sentiment analysis predictions on a given topic. Tweets are scraped from twitter and analysed to gather opinions given a specific search term. There are two main scripts, the console and the training environment.
- **train_models.py** : train new machine learning models on SemEval 2016 sentiment analysis data
- **console.py** : use existing models to to make sentiment analysis predictions - an image depicting the sentiment values per class is saved by the name *"Sentiment_of_searchterm.png"*

To run this repository, the following packages must be installed and run using Python3:
- spacy
- python-twitter
- tweepy
- pandas
- pickle
- sklearn
- numpy
- matplotlib

To use the console:
1. Clone the repository
2. From a terminal, navigate to the root directory
3. On windows, execute the command "python console.py"
4. If using linux, execute the command "python3 console.py"
