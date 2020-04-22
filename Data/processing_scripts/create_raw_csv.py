import re
import time
from twitter import *

# --------------------------------------------------------------------------------
# These tokens are needed for user authentication.
# Credentials can be generated via Twitter's Application Management:
#	https://developer.twitter.com/en/apps
# --------------------------------------------------------------------------------


# ----------------- CONFIGURE SETTINGS -----------------
settings = {
    "consumer_key": "LFUmZ6im9Nm7GDZ1yjjKOszpe",
    "consumer_secret": "XXRi6iKS5MtJyNSFhJt4ArqjWdv2vzdgkiLdlaEat2CbVV1Kh2",
    "access_key": "992471517891002368-vKRMWeuRPEaebFX0Z0w2pIozPdYSsIC",
    "access_secret": "oYWeCVgu3vnZtPRd9HyIvv1A1Wk44QsDDPPtpNoXHslNS"
}

api = Api(settings['consumer_key'], settings["consumer_secret"],
          access_token_key=settings["access_key"], access_token_secret=settings["access_secret"], sleep_on_rate_limit=True)

def get_tweets_by_id(id_string):
    try:
        tweet = api.GetStatus(id_string)
        return tweet.AsDict()['text']
    except TwitterError as e:
        print(e)
        return None


# ---------------------------------------------------------
def create_csv(text_file_name: str, rm_tweets: int) -> int:
    """
    :param text_file_name: Relative path to file
    :param rm_tweets: number of tweets removed
    :return: number of tweets removed in current file
    """
    # Open files in read mode
    csv = open("../processed_data/RawData.csv", "a", encoding="ISO-8859-1")
    text_file = open("../raw_data/" + text_file_name, "r", encoding="ISO-8859-1").readlines()

    for index in range(len(text_file)):
        print('Index: ', index)

        if index % 250 == 0 and index > 0:
            print("Number of tweets downloaded: ", index)
            progress = index / len(text_file)
            print("Progress: " + str(round(progress, 2)*100) + '%')
            time_so_far = (time.time() - start_time) / 60
            print("Time taken so far (minutes): ", round(time_so_far, 2))
            print('\n')

        line = text_file[index].strip()
        first_space = line.find("\t")
        last_space = line.rfind("\t")

        tweet_id = line[0:first_space]
        subject = line[first_space + 1: last_space]

        label = line[last_space + 1:]

        tweet_text = get_tweets_by_id(tweet_id)

        if tweet_text is None:
            rm_tweets += 1
            continue

        tweet_text = tweet_text.replace('"', "'").strip('\n')
        subject = subject.replace('"', "'").strip('\n')

        tweet_text = re.sub("\n", '', tweet_text)

        line = '\n"' + label + '","' + subject + '","' + tweet_text + '"'

        try:
            csv.write(line)
        except UnicodeEncodeError:
            rm_tweets += 1
            continue

    csv.close()
    return rm_tweets


if __name__ == "__main__":
    start_time = time.time()
    open("../processed_data/RawData.csv", 'w', encoding="ISO-8859-1").close()
    csv = open("../processed_data/RawData.csv", "a", encoding="ISO-8859-1")
    csv.write('label,subject,text')
    csv.close()

    removed_tweets = create_csv("twitter-2016test-CE.txt", 0)
    removed_tweets = create_csv("twitter-2016train-CE.txt", removed_tweets)
    print(str(removed_tweets) + ' tweets have been removed')
