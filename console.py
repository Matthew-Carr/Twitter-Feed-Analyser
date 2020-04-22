import tweepy
import pandas as pd
from Data.processing_scripts.train_vectorisers import Vectoriser
from Data.processing_scripts.clean_data import clean_string
import matplotlib.pyplot as plt
import numpy as np

#--------------------------------------------------------------------------------
# These tokens are needed for user authentication.
# Credentials can be generated via Twitter's Application Management:
#	https://developer.twitter.com/en/apps
#--------------------------------------------------------------------------------


# ----------------- CONFIGURE SETTINGS -----------------
consumer_key = "rZbM6d03OnUi5Y7LgBJFxRmnS"
consumer_secret = "Hxw4M30hRdGm4CHS6Bl9XPhi5o93dwfYHf1XMAuU97wVsfbHbn"
access_key = "1558559760-dQRoIBAbIhmHr1WEafIpwj5OCTjnRkyFdK1yyp6"
access_secret = "a0jINBhqFW4VUuxP7KEMdDJ0jgdo4TZCdtFthpzYARjWt"

auth = tweepy.auth.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)
api = tweepy.API(auth)
# ---------------------------------------------------------


def visualise_sentiment(results: np.ndarray, search_term: str):
    results = results
    total = len(results)
    labels = ['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive']

    counts = [0] * 5

    for x in results:
        counts[x + 2] += 1

    for ind in range(len(counts)):
        counts[ind] /= total

    index = 0.6 * np.arange(len(labels))
    bar_width = 0.2

    plt.bar(index, counts, bar_width,
            alpha=1,
            color='dimgrey',
            label='Tweet data')

    plt.ylabel('Proportion of tweets', fontsize=13)
    plt.xlabel('Sentiment Labels', fontsize=13)
    plt.xticks(index + bar_width, labels)
    plt.legend()
    plt.title('Results', fontweight='bold', fontsize=14)

    plt.tight_layout()
    plt.savefig('Sentiment_of_' +search_term + '.png')
    plt.show()


def get_trained_models(settings: dict) -> tuple:
    v_and_f = [settings['vector']] + settings['features']

    trained_model = pd.read_pickle('Data/trained_models/' + settings['model_name'] + '_with_' +
                                   '_'.join([settings['vector']] + settings['features']) + '.pckl')

    trained_vectorisers = [pd.read_pickle('Data/vectorisers/' + name + '_vectoriser.pckl') for name in v_and_f]

    return trained_model, Vectoriser(trained_vectorisers)


def get_tweets(model, vectoriser):
    while True:
        search_term = input('Enter a search term: ')
       # search_amount = input('Enter number of tweets required: ')

        search_results = api.search(q=search_term, count=100, result_type='mixed', lang="en", tweet_mode='extended')

        # search_results is an empty list ([]) if there are no results returned

        if search_results:
            original_tweets = [tweet.full_text for tweet in search_results]
            tweet_texts = [clean_string(tweet.full_text) for tweet in search_results]
            tweet_vectors = vectoriser.transform(tweet_texts)
            predictions = model.predict(tweet_vectors)

            visualise_sentiment(predictions, search_term)

            print('Showing sample of 10 tweets...')
            unique_tweets = list(set((tweet_texts[i], predictions[i]) for i in range(len(tweet_texts))))

            for i in range(min(len(unique_tweets), 10)):
                clean_tweet = unique_tweets[i][0]
                prediction = unique_tweets[i][1]
                print('Clean Tweet: ', clean_tweet)
                print('Prediction: ', prediction)
        else:
            print('There are no search results matching this query...')

        cont = input('Continue? y/n')
        if cont == 'n':
            break


if __name__ == '__main__':
    # data_file = pd.read_csv("Data/processed_data/RawData.csv", encoding="ISO-8859-1")
    # visualise_sentiment(data_file['label'])

    model_settings = {'model_name': 'logistic_regression', 'vector': 'tf_idf', 'features': []}
    t_model, t_vectoriser = get_trained_models(settings=model_settings)

    get_tweets(t_model, t_vectoriser)
