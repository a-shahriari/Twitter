# Data Module

import os
import warnings
import numpy as np
import pandas as pd

from tabulate import tabulate

warnings.simplefilter(action='ignore')


def data_ingest(): # import the raw data, process/manipulate, and save in proper format for future modelling/analysis

    file_out = './results/Tweets.csv'

    # check if the data was processed before

    if os.path.exists(file_out):

        print('\n - Loading')
        data = pd.read_csv(file_out, encoding='iso-8859-1')

    else:

        np.set_printoptions(linewidth=150, threshold=100)
        print('\n - Importing')

        # load the company, ticker symbols and tweets

        company = pd.read_csv(os.path.join('./data', 'Company.csv'))
        print('\n', tabulate(company.head(10), headers='keys', tablefmt='simple', showindex=False, numalign='right', stralign='left'))

        company_tweet = pd.read_csv(os.path.join('./data', 'Company_Tweet.csv'))
        print('\n', tabulate(company_tweet.head(10), headers='keys', tablefmt='simple', showindex=False, numalign='right', stralign='left'))

        tweet = pd.read_csv(os.path.join('./data', 'Tweet.csv'))
        print('\n', tabulate(tweet.head(10), headers='keys', tablefmt='simple', showindex=False, numalign='right', stralign='left'))

        ##############################################################################

        print('\n - Processing')

        # format the date and time

        tweet['post_date'] = pd.to_datetime(tweet['post_date'], unit='s')
        tweet['day_date'] = tweet['post_date'].apply(lambda x: x.date())

        # merge tweets and ticker symbols via tweet IDs

        tweets = pd.merge(left=company_tweet, right=tweet, how='left', left_on='tweet_id', right_on='tweet_id')
        print('\n', tabulate(tweets.head(10), headers='keys', tablefmt='simple', showindex=False, numalign='right', stralign='left'))

        # load market values

        stock = pd.read_csv("./data/CompanyValues.csv")
        stock['day_date'] = stock['day_date'].apply(lambda x: pd.to_datetime(x).date())

        print('\n', tabulate(stock.head(10), headers='keys', tablefmt='simple', showindex=False, numalign='right', stralign='left'))

        # cross reference the tweets and stock values

        data = pd.merge(left=tweets, right=stock, how='left', on=['ticker_symbol', 'day_date']).drop(columns=['day_date']).reset_index(drop=True)
        data.columns = ['Tweet', 'Ticker', 'Poster', 'Time', 'Body', 'Comments', 'Retweets', 'Likes', 'Close Value', 'Volume', 'Open Value', 'High Value', 'Low Value']

        data.index += 1
        data.index.name = 'ID'

        print('\n', tabulate(data.head(10), headers='keys', tablefmt='simple', showindex=True, numalign='right', stralign='left'))

        ##############################################################################

        print('\n - Saving')

        # save in comma separated values (csv) format for further analysis

        data = data.applymap(lambda x: x.encode('unicode_escape').decode('iso-8859-1') if isinstance(x, str) else x)
        data.to_csv(str(file_out), index=False, encoding='iso-8859-1')

    return data
