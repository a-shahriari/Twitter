# Modelling Module

import os
import warnings
import re
import pickle
import numpy as np
import pandas as pd
import multiprocessing
import spacy
import matplotlib.pyplot as plt

from pathlib import Path
from textwrap import wrap
from tabulate import tabulate
from scipy import stats as stat
from spacytextblob.spacytextblob import SpacyTextBlob
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.parsing.preprocessing import preprocess_string, strip_tags, strip_punctuation, strip_multiple_whitespaces
from gensim.parsing.preprocessing import strip_numeric, remove_stopwords, strip_short

warnings.simplefilter(action='ignore')


def scale(x):  # min-max scaler

    # normalize a list of numbers by subtracting form the minimum value and dividing to the difference of maximum and minimum values
    return (x - x.min()) / (x.max() - x.min())


def model_q1(data, verbose):  # can you provide any correlation between tweets and stock market value?

    # configure the handles to plot the charts

    file_out = Path('./results/Q1-Plot-1.png')

    fig = plt.figure(figsize=(12, 8))

    fig.suptitle('\n Tweets and Stock Market vs Time \n', fontsize=10, fontweight='bold')
    fig.subplots_adjust(wspace=0.2, hspace=0.5)

    ##############################################################################

    print('\n - Plotting')

    ax = fig.add_subplot(3, 2, 1)

    ticker_dates = data[['Ticker', 'Time', 'Close Value', 'Volume', 'Open Value', 'High Value', 'Low Value']]
    ticker_dates['Time'] = pd.to_datetime(ticker_dates['Time'])

    # calculate the stock value, exchange, and trend

    ticker_dates['Value'] = (ticker_dates['High Value'] + ticker_dates['Low Value']) / 2
    ticker_dates['Exchange'] = ticker_dates['Close Value'] * ticker_dates['Volume']
    ticker_dates['Trend'] = ticker_dates['Close Value'] - ticker_dates['Open Value']

    # count the posted tweet number for each day for each company

    tweet_counts = ticker_dates.groupby(['Ticker', ticker_dates['Time'].dt.to_period("M")]).size().reset_index(level=[0, 1])
    tweet_counts['Time'] = tweet_counts['Time'].apply(lambda x: pd.to_datetime(str(x)))

    # plot the sequences for all companies

    for ticker_symbol in data['Ticker'].unique():
        tweets = tweet_counts[tweet_counts['Ticker'] == ticker_symbol][['Time', 0]].set_index('Time')
        ax.plot(tweets, label=ticker_symbol)

    # configure the title, grids, and axis

    ax.set_title('Tweets vs Time', fontsize=6, fontweight='bold')
    ax.legend(loc='upper right', fontsize=6)

    ax.grid(True)
    ax.set_axisbelow(True)

    ax.xaxis.grid(color='gray', linestyle='dashed')
    ax.yaxis.grid(color='gray', linestyle='dashed')

    ax.tick_params(direction='in')

    ##############################################################################

    ax = fig.add_subplot(3, 2, 2)

    # calculate the mean stock value on monthly basis

    stock_value = ticker_dates.groupby(['Ticker', ticker_dates['Time'].dt.to_period("M")]).agg({'Value': 'mean'}).reset_index(level=[0, 1])
    stock_value['Time'] = stock_value['Time'].apply(lambda x: pd.to_datetime(str(x)))

    # plot the sequences for each companies

    for ticker_symbol in data['Ticker'].unique():
        value = stock_value[stock_value['Ticker'] == ticker_symbol][['Time', 'Value']].set_index('Time')
        ax.plot(value, label=ticker_symbol)

    ax.set_title('Stock Value vs Time', fontsize=6, fontweight='bold')
    ax.legend(loc='upper right', fontsize=6)

    ax.grid(True)
    ax.set_axisbelow(True)

    ax.xaxis.grid(color='gray', linestyle='dashed')
    ax.yaxis.grid(color='gray', linestyle='dashed')

    ax.tick_params(direction='in')

    ##############################################################################

    ax = fig.add_subplot(3, 2, 3)

    # calculate the mean stock volume on monthly basis

    stock_volume = ticker_dates.groupby(['Ticker', ticker_dates['Time'].dt.to_period("M")]).agg({'Volume': 'mean'}).reset_index(level=[0, 1])
    stock_volume['Time'] = stock_volume['Time'].apply(lambda x: pd.to_datetime(str(x)))

    # plot the sequences for all companies

    for ticker_symbol in data['Ticker'].unique():
        volume = stock_volume[stock_volume['Ticker'] == ticker_symbol][['Time', 'Volume']].set_index('Time')
        ax.plot(volume, label=ticker_symbol)

    ax.set_title('Stock Volume vs Time', fontsize=6, fontweight='bold')
    ax.legend(loc='upper right', fontsize=6)

    ax.grid(True)
    ax.set_axisbelow(True)

    ax.xaxis.grid(color='gray', linestyle='dashed')
    ax.yaxis.grid(color='gray', linestyle='dashed')

    ax.tick_params(direction='in')

    ##############################################################################

    ax = fig.add_subplot(3, 2, 4)

    # calculate the mean stock exchange on monthly basis

    stock_exchange = ticker_dates.groupby(['Ticker', ticker_dates['Time'].dt.to_period("M")]).agg({'Exchange': 'mean'}).reset_index(level=[0, 1])
    stock_exchange['Time'] = stock_exchange['Time'].apply(lambda x: pd.to_datetime(str(x)))

    # plot the sequences for all companies

    for ticker_symbol in data['Ticker'].unique():
        exchange = stock_exchange[stock_exchange['Ticker'] == ticker_symbol][['Time', 'Exchange']].set_index('Time')
        ax.plot(exchange, label=ticker_symbol)

    ax.set_title('Stock Exchange vs Time', fontsize=6, fontweight='bold')
    ax.legend(loc='upper right', fontsize=6)

    ax.grid(True)
    ax.set_axisbelow(True)

    ax.xaxis.grid(color='gray', linestyle='dashed')
    ax.yaxis.grid(color='gray', linestyle='dashed')

    ax.tick_params(direction='in')

    # save the figure

    plt.savefig(str(file_out), dpi=300)

    ##############################################################################
    ax = fig.add_subplot(3, 2, 5)

    # calculate the mean stock trend for each company

    stock_trend = ticker_dates.groupby(['Ticker', ticker_dates['Time'].dt.to_period("M")]).agg({'Trend': 'mean'}).reset_index(level=[0, 1])
    stock_trend['Time'] = stock_trend['Time'].apply(lambda x: pd.to_datetime(str(x)))

    # plot the sequences for all companies

    for ticker_symbol in data['Ticker'].unique():
        trend = stock_trend[stock_trend['Ticker'] == ticker_symbol][['Time', 'Trend']].set_index('Time')
        ax.plot(trend, label=ticker_symbol)

    ax.set_title('Stock Trend vs Time', fontsize=6, fontweight='bold')
    ax.legend(loc='upper right', fontsize=6)

    ax.grid(True)
    ax.set_axisbelow(True)

    ax.xaxis.grid(color='gray', linestyle='dashed')
    ax.yaxis.grid(color='gray', linestyle='dashed')

    ax.tick_params(direction='in')

    plt.savefig(str(file_out), dpi=300)

    ##############################################################################
    ##############################################################################

    file_out = Path('./results/Q1-Plot-2.png')

    fig = plt.figure(figsize=(12, 8))

    fig.suptitle('\n Top Companies vs Tweets and Stock Market \n', fontsize=10, fontweight='bold')
    fig.subplots_adjust(wspace=0.2, hspace=0.5)

    # scale the number of tweets and stock variables (value, volume, exchange, trend) and plot them in one chart for each company

    for ticker_index, ticker_symbol in enumerate(data['Ticker'].unique()):
        ax = fig.add_subplot(3, 2, ticker_index + 1)

        ax.plot(scale(tweet_counts[tweet_counts['Ticker'] == ticker_symbol][['Time', 0]].set_index('Time')), label='Tweets')
        ax.plot(scale(stock_value[stock_value['Ticker'] == ticker_symbol][['Time', 'Value']].set_index('Time')), label='Value')
        ax.plot(scale(stock_volume[stock_volume['Ticker'] == ticker_symbol][['Time', 'Volume']].set_index('Time')), label='Volume')
        ax.plot(scale(stock_exchange[stock_exchange['Ticker'] == ticker_symbol][['Time', 'Exchange']].set_index('Time')), label='Exchange')
        ax.plot(scale(stock_trend[stock_trend['Ticker'] == ticker_symbol][['Time', 'Trend']].set_index('Time')), label='Trend')

        ax.set_title(ticker_symbol, fontsize=6, fontweight='bold')
        ax.legend(loc='upper right', fontsize=6)

        ax.grid(True)
        ax.set_axisbelow(True)

        ax.xaxis.grid(color='gray', linestyle='dashed')
        ax.yaxis.grid(color='gray', linestyle='dashed')

        ax.tick_params(direction='in')

    plt.savefig(str(file_out), dpi=300)

    ##############################################################################
    ##############################################################################

    for ticker_index, ticker_symbol in enumerate(data['Ticker'].unique()):
        file_out = Path('./results/Q1-Plot-' + ticker_symbol + '.png')

        fig = plt.figure(figsize=(12, 8))

        fig.suptitle('\n {} - Tweets vs Stock Market \n'.format(ticker_symbol), fontsize=10, fontweight='bold')
        fig.subplots_adjust(wspace=0.2, hspace=0.5)

        # scale and create the time sequence of tweets and stock variables

        tweets = scale(tweet_counts[tweet_counts['Ticker'] == ticker_symbol][['Time', 0]].set_index('Time'))
        value = scale(stock_value[stock_value['Ticker'] == ticker_symbol][['Time', 'Value']].set_index('Time'))
        volume = scale(stock_volume[stock_volume['Ticker'] == ticker_symbol][['Time', 'Volume']].set_index('Time'))
        exchange = scale(stock_exchange[stock_exchange['Ticker'] == ticker_symbol][['Time', 'Exchange']].set_index('Time'))
        trend = scale(stock_trend[stock_trend['Ticker'] == ticker_symbol][['Time', 'Trend']].set_index('Time'))

        ##############################################################################

        ax = fig.add_subplot(3, 2, 1)

        # plot all the scaled sequences in one chart

        ax.plot(tweets, label='Tweets')
        ax.plot(value, label='Value')
        ax.plot(volume, label='Volume')
        ax.plot(exchange, label='Exchange')
        ax.plot(trend, label='Trend')

        ax.set_title(ticker_symbol, fontsize=6, fontweight='bold')
        ax.legend(loc='upper right', fontsize=6)

        ax.grid(True)
        ax.set_axisbelow(True)

        ax.xaxis.grid(color='gray', linestyle='dashed')
        ax.yaxis.grid(color='gray', linestyle='dashed')

        ax.tick_params(direction='in')

        ##############################################################################

        ax = fig.add_subplot(3, 2, 3)

        # plot the scaterr of tweets vs stock values

        ax.scatter(tweets, value, label='Value')
        ax.plot(np.arange(0.0, 1, 0.01), np.arange(0.0, 1, 0.01), linestyle='--', linewidth=1)

        ax.set_title('Tweets vs Value', fontsize=6, fontweight='bold')

        ax.grid(True)
        ax.set_axisbelow(True)

        ax.xaxis.grid(color='gray', linestyle='dashed')
        ax.yaxis.grid(color='gray', linestyle='dashed')

        ax.tick_params(direction='in')

        ##############################################################################

        ax = fig.add_subplot(3, 2, 4)

        # plot the scaterr of tweets vs stock volume

        ax.scatter(tweets, volume, label='Volume')
        ax.plot(np.arange(0.0, 1, 0.01), np.arange(0.0, 1, 0.01), linestyle='--', linewidth=1)

        ax.set_title('Tweets vs Volume', fontsize=6, fontweight='bold')

        ax.grid(True)
        ax.set_axisbelow(True)

        ax.xaxis.grid(color='gray', linestyle='dashed')
        ax.yaxis.grid(color='gray', linestyle='dashed')

        ax.tick_params(direction='in')

        ##############################################################################

        ax = fig.add_subplot(3, 2, 5)

        # plot the scaterr of tweets vs stock exchange

        ax.scatter(tweets, exchange, label='Exchange')
        ax.plot(np.arange(0.0, 1, 0.01), np.arange(0.0, 1, 0.01), linestyle='--', linewidth=1)

        ax.set_title('Tweets vs Exchange', fontsize=6, fontweight='bold')

        ax.grid(True)
        ax.set_axisbelow(True)

        ax.xaxis.grid(color='gray', linestyle='dashed')
        ax.yaxis.grid(color='gray', linestyle='dashed')

        ax.tick_params(direction='in')

        ##############################################################################

        ax = fig.add_subplot(3, 2, 6)

        # plot the scaterr of tweets vs stock trend

        ax.scatter(tweets, trend, label='Trend')
        ax.plot(np.arange(0.0, 1, 0.01), np.arange(0.0, 1, 0.01), linestyle='--', linewidth=1)

        ax.set_title('Tweets vs Trend', fontsize=6, fontweight='bold')

        ax.grid(True)
        ax.set_axisbelow(True)

        ax.xaxis.grid(color='gray', linestyle='dashed')
        ax.yaxis.grid(color='gray', linestyle='dashed')

        ax.tick_params(direction='in')

        plt.savefig(str(file_out), dpi=300)

    ##############################################################################
    ##############################################################################

    print('\n - Correlation Analysis')

    # calculate and print the correlations and p-values for tweets vs stock variables

    df_corr = pd.DataFrame()

    for ticker_index, ticker_symbol in enumerate(data['Ticker'].unique()):
        tweets = tweet_counts[tweet_counts['Ticker'] == ticker_symbol][['Time', 0]].set_index('Time')

        value = stock_value[stock_value['Ticker'] == ticker_symbol][['Time', 'Value']].set_index('Time')
        corr, pval = stat.spearmanr(tweets, value, nan_policy='omit')
        df_corr = df_corr.append({'Ticker': ticker_symbol, 'Stock': 'Value', 'Correlation': corr, 'P-Value': pval}, ignore_index=True)

        volume = stock_volume[stock_volume['Ticker'] == ticker_symbol][['Time', 'Volume']].set_index('Time')
        corr, pval = stat.spearmanr(tweets, volume, nan_policy='omit')
        df_corr = df_corr.append({'Ticker': ticker_symbol, 'Stock': 'Volume', 'Correlation': corr, 'P-Value': pval}, ignore_index=True)

        exchange = stock_exchange[stock_exchange['Ticker'] == ticker_symbol][['Time', 'Exchange']].set_index('Time')
        corr, pval = stat.spearmanr(tweets, exchange, nan_policy='omit')
        df_corr = df_corr.append({'Ticker': ticker_symbol, 'Stock': 'Exchange', 'Correlation': corr, 'P-Value': pval}, ignore_index=True)

        trend = stock_trend[stock_trend['Ticker'] == ticker_symbol][['Time', 'Trend']].set_index('Time')
        corr, pval = stat.spearmanr(tweets, trend, nan_policy='omit')
        df_corr = df_corr.append({'Ticker': ticker_symbol, 'Stock': 'Trend', 'Correlation': corr, 'P-Value': pval}, ignore_index=True)

    df_corr = df_corr[['Ticker', 'Stock', 'Correlation', 'P-Value']]

    df_corr.index += 1
    df_corr.index.name = 'ID'

    print('\n', tabulate(df_corr, headers='keys', tablefmt='simple', showindex=False, numalign='right', stralign='left'))

    return


def model_q2(data, verbose):  # is negative or positive sentiment more effective in influencing market value of a share?
    print('\n - Sentiment Analysis')

    # configure the Spacy library for the sentiment analysis

    nlp = spacy.load('en_core_web_sm')
    spacy_text_blob = SpacyTextBlob()
    nlp.add_pipe(spacy_text_blob)

    ticker_dates = data[['Tweet', 'Ticker', 'Poster', 'Body', 'Time', 'Comments', 'Retweets', 'Likes']]
    ticker_dates['Time'] = pd.to_datetime(ticker_dates['Time'])

    # calculate the overall number of comments, retweets, and likes for each poster and company

    poster_engages = ticker_dates.groupby(['Ticker', 'Poster']).agg({'Comments': 'sum', 'Retweets': 'sum', 'Likes': 'sum'}).reset_index(level=[0, 1])

    # calculate the overall interaction

    poster_engages['Interacts'] = poster_engages['Comments'] + poster_engages['Retweets'] + poster_engages['Likes']

    poster_engages.columns = ['Ticker', 'Poster', 'Comments', 'Retweets', 'Likes', 'Interacts']
    poster_engages = poster_engages[poster_engages['Interacts'] > 0].sort_values(by=['Interacts'], ascending=False).reset_index(drop=True)

    # filter the influential posters by the number of their interactions

    df_tweets = data[data['Poster'].isin(list(poster_engages[poster_engages['Interacts'] > 1e4]['Poster']))].reset_index(drop=True)

    # check if the sentiments were extracted before

    file_out = Path('./results/repository/Sentiments.csv')

    if os.path.exists(file_out):

        print('\n - Loading')
        df_sentiments = pd.read_csv(file_out, encoding='iso-8859-1')

    else:

        df_sentiments = pd.DataFrame()

        for idx, row in df_tweets.iterrows():

            tweet = str(row['Tweet'])
            ticker = row['Ticker']
            time = row['Time']
            poster = row['Poster']
            body = str(row['Body'])

            # calculate the polarity and subjectivity of each tweets

            doc = nlp(body)

            polarity = doc._.sentiment.polarity
            subjectivity = doc._.sentiment.subjectivity

            df_sentiments = df_sentiments.append({'Tweet': tweet, 'Ticker': ticker, 'Time': time, 'Poster': poster,
                                                  'Body': body, 'Polarity': round(polarity, 2), 'Subjectivity': round(subjectivity, 2)}, ignore_index=True).reset_index(drop=True)

            # format and print the outcome of sentiment analysis

            if verbose:
                np.set_printoptions(linewidth=150, threshold=100)

                print('\n\n REQ {}-{} \n CASE: {} \n BODY: {} \n POLARITY: {} \n SUBJECTIVITY: {}'.format(
                    str(idx + 1), str(len(df_tweets)), str(tweet),
                    "\n".join(wrap(body, 100)), '{:.2f}'.format(polarity), '{:.2f}'.format(subjectivity)))

        df_sentiments = df_sentiments[['Tweet', 'Ticker', 'Poster', 'Time', 'Body', 'Polarity', 'Subjectivity']]

        df_sentiments.index += 1
        df_sentiments.index.name = 'ID'

        print('\n', tabulate(df_sentiments.head(10), headers='keys', tablefmt='simple', showindex=True, numalign='right', stralign='left'))

        # save the sentiments for future analysis

        df_sentiments = df_sentiments.applymap(lambda x: x.encode('unicode_escape').decode('iso-8859-1') if isinstance(x, str) else x)
        df_sentiments.to_csv(str(file_out), index=False, encoding='iso-8859-1')

    ##############################################################################

    # join the tweets with sentiments by the tweet ID

    df_tweets = pd.merge(left=df_sentiments, right=data, how='left', left_on='Tweet', right_on='Tweet')

    df_tweets = df_tweets[['Tweet', 'Ticker_x', 'Poster_x', 'Time_x', 'Body_x', 'Polarity', 'Subjectivity', 'Close Value', 'Volume', 'Open Value', 'High Value', 'Low Value']]
    df_tweets.columns = ['Tweet', 'Ticker', 'Poster', 'Time', 'Body', 'Polarity', 'Subjectivity', 'Close Value', 'Volume', 'Open Value', 'High Value', 'Low Value']

    print('\n', tabulate(df_tweets.head(10), headers='keys', tablefmt='simple', showindex=True, numalign='right', stralign='left'))

    # calculate the stock varaibales

    ticker_dates = df_tweets[['Ticker', 'Time', 'Polarity', 'Close Value', 'Volume', 'Open Value', 'High Value', 'Low Value']]
    ticker_dates['Time'] = pd.to_datetime(ticker_dates['Time'])

    ticker_dates['Value'] = (ticker_dates['High Value'] + ticker_dates['Low Value']) / 2
    ticker_dates['Exchange'] = ticker_dates['Close Value'] * ticker_dates['Volume']
    ticker_dates['Trend'] = ticker_dates['Close Value'] - ticker_dates['Open Value']

    ##############################################################################
    ##############################################################################

    # process overall, positive, and negative sentiments

    for sent in ['ALL', 'POS', 'NEG']:

        if sent == 'ALL':
            print('\n - Overall Correlation Analysis')

            # filter out neutral sentiments
            tickers = ticker_dates[ticker_dates['Polarity'] != 0]

        elif sent == 'POS':
            print('\n - Positive Correlation Analysis')

            # filter positive sentiments
            tickers = ticker_dates[ticker_dates['Polarity'] > 0]

        elif sent == 'NEG':
            print('\n - Negative Correlation Analysis')

            # filter negative sentiments
            tickers = ticker_dates[ticker_dates['Polarity'] < 0]

        # calculate mean polarity and sock variables on monthly basis

        tweet_sents = tickers.groupby(['Ticker', tickers['Time'].dt.to_period("M")]).agg({'Polarity': 'mean'}).reset_index(level=[0, 1])
        tweet_sents['Time'] = tweet_sents['Time'].apply(lambda x: pd.to_datetime(str(x)))

        stock_value = tickers.groupby(['Ticker', tickers['Time'].dt.to_period("M")]).agg({'Value': 'mean'}).reset_index(level=[0, 1])
        stock_value['Time'] = stock_value['Time'].apply(lambda x: pd.to_datetime(str(x)))

        stock_volume = tickers.groupby(['Ticker', tickers['Time'].dt.to_period("M")]).agg({'Volume': 'mean'}).reset_index(level=[0, 1])
        stock_volume['Time'] = stock_volume['Time'].apply(lambda x: pd.to_datetime(str(x)))

        stock_exchange = tickers.groupby(['Ticker', tickers['Time'].dt.to_period("M")]).agg({'Exchange': 'mean'}).reset_index(level=[0, 1])
        stock_exchange['Time'] = stock_exchange['Time'].apply(lambda x: pd.to_datetime(str(x)))

        stock_trend = tickers.groupby(['Ticker', tickers['Time'].dt.to_period("M")]).agg({'Trend': 'mean'}).reset_index(level=[0, 1])
        stock_trend['Time'] = stock_trend['Time'].apply(lambda x: pd.to_datetime(str(x)))

        ##############################################################################

        df_corr = pd.DataFrame()

        for ticker_index, ticker_symbol in enumerate(data['Ticker'].unique()):
            file_out = Path('./results/Q2-' + sent + '-Plot-' + ticker_symbol + '.png')

            fig = plt.figure(figsize=(12, 8))

            fig.suptitle('\n {} - Sentiments vs Stock Market \n'.format(ticker_symbol), fontsize=10, fontweight='bold')
            fig.subplots_adjust(wspace=0.2, hspace=0.5)

            # create the sequence of tweets and stock variables

            tweets = scale(tweet_sents[tweet_sents['Ticker'] == ticker_symbol][['Time', 'Polarity']].set_index('Time'))
            value = scale(stock_value[stock_value['Ticker'] == ticker_symbol][['Time', 'Value']].set_index('Time'))
            volume = scale(stock_volume[stock_volume['Ticker'] == ticker_symbol][['Time', 'Volume']].set_index('Time'))
            exchange = scale(stock_exchange[stock_exchange['Ticker'] == ticker_symbol][['Time', 'Exchange']].set_index('Time'))
            trend = scale(stock_trend[stock_trend['Ticker'] == ticker_symbol][['Time', 'Trend']].set_index('Time'))

            ##############################################################################

            ax = fig.add_subplot(3, 2, 1)

            # plot the sequences

            ax.plot(tweets, label='Sentiments')
            ax.plot(value, label='Value')
            ax.plot(volume, label='Volume')
            ax.plot(exchange, label='Exchange')
            ax.plot(trend, label='Trend')

            ax.set_title(ticker_symbol, fontsize=6, fontweight='bold')
            ax.legend(loc='upper right', fontsize=6)

            ax.grid(True)
            ax.set_axisbelow(True)

            ax.xaxis.grid(color='gray', linestyle='dashed')
            ax.yaxis.grid(color='gray', linestyle='dashed')

            ax.tick_params(direction='in')

            ##############################################################################

            ax = fig.add_subplot(3, 2, 3)

            # plot scatter of sentiments vs sock values

            ax.scatter(tweets, value, label='Value')
            ax.plot(np.arange(0.0, 1, 0.01), np.arange(0.0, 1, 0.01), linestyle='--', linewidth=1)

            ax.set_title('Sentiments vs Value', fontsize=6, fontweight='bold')

            ax.grid(True)
            ax.set_axisbelow(True)

            ax.xaxis.grid(color='gray', linestyle='dashed')
            ax.yaxis.grid(color='gray', linestyle='dashed')

            ax.tick_params(direction='in')

            ##############################################################################

            ax = fig.add_subplot(3, 2, 4)

            # plot scatter of sentiments vs sock volume

            ax.scatter(tweets, volume, label='Volume')
            ax.plot(np.arange(0.0, 1, 0.01), np.arange(0.0, 1, 0.01), linestyle='--', linewidth=1)

            ax.set_title('Sentiments vs Volume', fontsize=6, fontweight='bold')

            ax.grid(True)
            ax.set_axisbelow(True)

            ax.xaxis.grid(color='gray', linestyle='dashed')
            ax.yaxis.grid(color='gray', linestyle='dashed')

            ax.tick_params(direction='in')

            ##############################################################################

            ax = fig.add_subplot(3, 2, 5)

            # plot scatter of sentiments vs sock exchange

            ax.scatter(tweets, exchange, label='Exchange')
            ax.plot(np.arange(0.0, 1, 0.01), np.arange(0.0, 1, 0.01), linestyle='--', linewidth=1)

            ax.set_title('Sentiments vs Exchange', fontsize=6, fontweight='bold')

            ax.grid(True)
            ax.set_axisbelow(True)

            ax.xaxis.grid(color='gray', linestyle='dashed')
            ax.yaxis.grid(color='gray', linestyle='dashed')

            ax.tick_params(direction='in')

            ##############################################################################

            ax = fig.add_subplot(3, 2, 6)

            # plot scatter of sentiments vs sock trends

            ax.scatter(tweets, trend, label='Trend')
            ax.plot(np.arange(0.0, 1, 0.01), np.arange(0.0, 1, 0.01), linestyle='--', linewidth=1)

            ax.set_title('Sentiments vs Trend', fontsize=6, fontweight='bold')

            ax.grid(True)
            ax.set_axisbelow(True)

            ax.xaxis.grid(color='gray', linestyle='dashed')
            ax.yaxis.grid(color='gray', linestyle='dashed')

            ax.tick_params(direction='in')

            plt.savefig(str(file_out), dpi=300)

            ##############################################################################
            ##############################################################################

            # calculate the correlations of sentiments and stock variables

            corr, pval = stat.spearmanr(tweets, value, nan_policy='omit')
            df_corr = df_corr.append({'Ticker': ticker_symbol, 'Stock': 'Value', 'Correlation': corr, 'P-Value': pval}, ignore_index=True)

            corr, pval = stat.spearmanr(tweets, volume, nan_policy='omit')
            df_corr = df_corr.append({'Ticker': ticker_symbol, 'Stock': 'Volume', 'Correlation': corr, 'P-Value': pval}, ignore_index=True)

            corr, pval = stat.spearmanr(tweets, exchange, nan_policy='omit')
            df_corr = df_corr.append({'Ticker': ticker_symbol, 'Stock': 'Exchange', 'Correlation': corr, 'P-Value': pval}, ignore_index=True)

            corr, pval = stat.spearmanr(tweets, trend, nan_policy='omit')
            df_corr = df_corr.append({'Ticker': ticker_symbol, 'Stock': 'Trend', 'Correlation': corr, 'P-Value': pval}, ignore_index=True)

        # format and print the correlations for all companies

        df_corr = df_corr[['Ticker', 'Stock', 'Correlation', 'P-Value']]

        df_corr.index += 1
        df_corr.index.name = 'ID'

        print('\n', tabulate(df_corr, headers='keys', tablefmt='simple', showindex=False, numalign='right', stralign='left'))

    return


def model_q3(data, verbose):  # which posters (if any) are most influential in influencing stock market value and why/how are they more influential?
    print('\n Engagement Analysis')

    # calculate the overall number of interactions for each poster and company

    ticker_dates = data[['Ticker', 'Poster', 'Time', 'Comments', 'Retweets', 'Likes']]
    ticker_dates['Time'] = pd.to_datetime(ticker_dates['Time'])

    poster_engages = ticker_dates.groupby(['Ticker', 'Poster']).agg({'Comments': 'sum', 'Retweets': 'sum', 'Likes': 'sum'}).reset_index(level=[0, 1])
    poster_engages['Interacts'] = poster_engages['Comments'] + poster_engages['Retweets'] + poster_engages['Likes']

    poster_engages.columns = ['Ticker', 'Poster', 'Comments', 'Retweets', 'Likes', 'Interacts']
    poster_engages = poster_engages[poster_engages['Interacts'] > 0].sort_values(by=['Interacts'], ascending=False).reset_index(drop=True)

    poster_engages.index += 1
    poster_engages.index.name = 'ID'

    print('\n', tabulate(poster_engages.head(50), headers='keys', tablefmt='simple', showindex=True, numalign='right', stralign='left'))

    ##############################################################################

    # calculate the overall number of comments for each poster and company

    poster_comments = ticker_dates.groupby(['Ticker', 'Poster']).agg({'Comments': 'sum'}).reset_index(level=[0, 1])
    poster_comments.columns = ['Ticker', 'Poster', 'Comments']

    poster_comments = poster_comments[poster_comments['Comments'] > 0].sort_values(by=['Comments'], ascending=False).reset_index(drop=True)

    poster_comments.index += 1
    poster_comments.index.name = 'ID'

    print('\n', tabulate(poster_comments.head(50), headers='keys', tablefmt='simple', showindex=True, numalign='right', stralign='left'))

    ##############################################################################

    # calculate the overall number of retweets for each poster and company

    poster_retweets = ticker_dates.groupby(['Ticker', 'Poster']).agg({'Retweets': 'sum'}).reset_index(level=[0, 1])
    poster_retweets.columns = ['Ticker', 'Poster', 'Retweets']

    poster_retweets = poster_retweets[poster_retweets['Retweets'] > 0].sort_values(by=['Retweets'], ascending=False).reset_index(drop=True)

    poster_retweets.index += 1
    poster_retweets.index.name = 'ID'

    print('\n', tabulate(poster_retweets.head(50), headers='keys', tablefmt='simple', showindex=True, numalign='right', stralign='left'))

    ##############################################################################

    # calculate the overall number of likes for each poster and company

    poster_likes = ticker_dates.groupby(['Ticker', 'Poster']).agg({'Likes': 'sum'}).reset_index(level=[0, 1])
    poster_likes.columns = ['Ticker', 'Poster', 'Likes']

    poster_likes = poster_likes[poster_likes['Likes'] > 0].sort_values(by=['Likes'], ascending=False).reset_index(drop=True)

    poster_likes.index += 1
    poster_likes.index.name = 'ID'

    print('\n', tabulate(poster_likes.head(50), headers='keys', tablefmt='simple', showindex=True, numalign='right', stralign='left'))

    return


def hashfn(x):  # regularize the randomization for the language modelling for the sake of future replications
    return ord(x[0])


def parse(text):  # pre-process the text to prune tags, punctuations, whitespaces, stop/short words, and numbers

    # filter the line breaks

    if not len(str(text)):
        return str(' ')
    else:
        body = str(text).replace('\n', ' ')

    # prune the text

    filters = [lambda x: x.lower(), lambda x: re.sub('[^a-zA-Z]+', ' ', x),
               strip_tags, strip_punctuation, strip_multiple_whitespaces, remove_stopwords, strip_numeric, strip_short]

    body = ' '.join(preprocess_string(body, filters))

    return body


def model_q4(data, verbose):  # do you suspect any of the posters (using different handles) are related and why?
    print('\n Language Model Developing')
    np.random.seed(88)

    model_path: str = str.lower('./results/repository/models/embedding.gsm')

    if not os.path.exists('./results/repository/models'):
        os.makedirs('./results/repository/models')

    # check if the trained language model is available

    if os.path.exists(model_path):

        print('\n - Loading Language Model ')
        model = Doc2Vec.load(model_path)

        print('\n\t {}'.format(model).expandtabs(2))

    else:

        print('\n - Document Tagging')

        # build the document repository

        df_tweets = data.copy()
        df_docs = [TaggedDocument(parse(str(doc)), [idx]) for idx, doc in enumerate(df_tweets['Body'])]

        #########################################################################

        print('\n\n - Language Modeling')

        # configure the model hyper-parameters

        model = Doc2Vec(vector_size=100, dm=1, window=5, min_count=1, alpha=1e-2, min_alpha=1e-3,
                        epochs=10, seed=88, workers=multiprocessing.cpu_count(), hs=0, negative=5, hashfxn=hashfn)

        print('\n\t - Fields: {} \n\t - Instances: {} \n\n\t - Vector: {} \n\t - Window: {} \n\t - Rate: {} \n\t - Epochs: {} \n\t - Workers: {}'.format(
            str(len(df_tweets.columns)), str(len(df_tweets)), str(model.vector_size), str(model.window), str(model.alpha), str(model.epochs), str(model.workers)).expandtabs(2))

        # build the vocabulary and train the model

        model.build_vocab(df_docs)
        model.train(df_docs, total_examples=model.corpus_count, epochs=model.epochs)

        # save the model

        model.save(model_path)

        # save the parameters in pickle format

        params_path = './results/repository/models/params.pkl'

        params = {'Fields': len(df_tweets.columns), 'Instances': len(df_tweets),
                  'Vector': model.vector_size, 'Window': model.window, 'Rate': model.alpha, 'Epochs': model.epochs, 'Workers': model.workers}

        f = open(params_path, 'wb')
        pickle.dump(params, f)
        f.close()

        #########################################################################

        print('\n - Language Inference')

        file_out = Path('./results/repository/Embeddings.csv')

        df_embeddings = pd.DataFrame()

        for idx, row in df_tweets.iterrows():

            tweet = row['Tweet']
            ticker = row['Ticker']
            time = row['Time']
            poster = row['Poster']
            body = str(row['Body'])

            # pre=process the tweet

            doc = parse(body)

            if not isinstance(doc, str):
                continue
            elif len(doc.split()) < 1:
                continue

            # calculate the feature vector (embedding) by the language model

            model.random.seed(88)
            embedding = model.infer_vector(doc.split(), steps=10, alpha=1e-2)

            df_embeddings = df_embeddings.append({'Tweet': tweet, 'ticker': ticker, 'Time': time, 'Poster': poster,
                                                  'Body': body, 'Token': doc, 'Embedding': embedding}, ignore_index=True).reset_index(drop=True)

            # print the outcomes

            if verbose:
                np.set_printoptions(linewidth=150, threshold=100)

                print('\n\n REQ {}-{} \n CASE: {} \n BODY: {} \n TOKEN: {} \n EMBEDDING: {}'.format(
                    str(idx + 1), str(len(df_tweets)), str(tweet),
                    "\n".join(wrap(body, 100)), "\n".join(wrap(doc, 100)), embedding))

        # save the embeddings in csv format

        df_embeddings = df_embeddings.applymap(lambda x: x.encode('unicode_escape').decode('iso-8859-1') if isinstance(x, str) else x)
        df_embeddings.to_csv(str(file_out), index=False, encoding='iso-8859-1')

        print('\n\t {}'.format(model).expandtabs(2))

    ##############################################################################

    print('\n - Similarity Analysis ')

    file_out = Path('./results/repository/Posts.csv')

    # check if the tweets were joint as posts before

    if os.path.exists(file_out):

        print('\n - Loading')
        df_posts = pd.read_csv(file_out, encoding='iso-8859-1')

    else:

        # find the influential posters

        ticker_dates = data[['Tweet', 'Ticker', 'Poster', 'Body', 'Time', 'Comments', 'Retweets', 'Likes']]
        ticker_dates['Time'] = pd.to_datetime(ticker_dates['Time'])

        poster_engages = ticker_dates.groupby(['Ticker', 'Poster']).agg({'Comments': 'sum', 'Retweets': 'sum', 'Likes': 'sum'}).reset_index(level=[0, 1])
        poster_engages['Interacts'] = poster_engages['Comments'] + poster_engages['Retweets'] + poster_engages['Likes']

        poster_engages.columns = ['Ticker', 'Poster', 'Comments', 'Retweets', 'Likes', 'Interacts']
        poster_engages = poster_engages[poster_engages['Interacts'] > 1e4].sort_values(by=['Interacts'], ascending=False).reset_index(drop=True)

        # join all the tweets of influential posters as a single post

        df_posts = data[data['Poster'].isin(list(poster_engages['Poster']))].groupby('Poster')['Body'].apply(' '.join).reset_index()
        df_posts.columns = ['Poster', 'Post']

        # save the aggregated tweets (posts)

        df_posts = df_posts.applymap(lambda x: x.encode('unicode_escape').decode('iso-8859-1') if isinstance(x, str) else x)
        df_posts.to_csv(str(file_out), index=False, encoding='iso-8859-1')

    ##############################################################################

    file_out = Path('./results/repository/Scores.csv')

    posters = list(df_posts['Poster'])
    posts = list(df_posts['Post'])

    # calculate the similarity scores among each pair of influential posters

    score = [[0 for x in range(len(posters))] for y in range(len(posters))]

    df_scores = pd.DataFrame()

    for idx1, post1 in enumerate(posts):
        for idx2, post2 in enumerate(posts):
            score[idx1][idx2] = model.n_similarity(parse(post1), parse(post2))
            df_scores = df_scores.append({'Poster1': posters[idx1], 'Poster2': posters[idx2], 'Similarity': score[idx1][idx2]}, ignore_index=True)

    # filter the similarity scores

    df_scores = df_scores[['Poster1', 'Poster2', 'Similarity']]
    df_scores = df_scores[(df_scores['Similarity'] > 0.9970) & (df_scores['Similarity'] < 1)]

    # join all the counterparts to make a single row

    df_scores = df_scores.groupby('Poster1')['Poster2'].apply(', '.join).reset_index()
    df_scores.columns = ['Poster', 'Similar Posters']

    # sort the posters by the numbe rof their counterparts

    df_scores = df_scores.reindex(df_scores['Similar Posters'].str.len().sort_values(ascending=False).index).reset_index(drop=True)

    df_scores.index += 1
    df_scores.index.name = 'ID'

    print('\n', tabulate(df_scores, headers='keys', tablefmt='simple', showindex=True, numalign='right', stralign='left'))

    # save the posters and their counterparts

    df_scores = df_scores.applymap(lambda x: x.encode('unicode_escape').decode('iso-8859-1') if isinstance(x, str) else x)
    df_scores.to_csv(str(file_out), index=False, encoding='iso-8859-1')

    ##############################################################################

    print('\n - Plotting')

    file_out = Path('./results/Q4-Plot.png')

    fig = plt.figure(figsize=(12, 8))

    fig.suptitle('\n Posters \n', fontsize=10, fontweight='bold')
    fig.subplots_adjust(wspace=0.2, hspace=0.5)

    # plot the colour map of similarity scores

    ax = fig.add_subplot(111)
    plt.imshow(score)
    ax.set_aspect('equal')

    ax.set_title('Post Similarity', fontsize=6, fontweight='bold')

    ax.grid(True)
    ax.set_axisbelow(True)

    ax.xaxis.grid(color='gray', linestyle='dashed')
    ax.yaxis.grid(color='gray', linestyle='dashed')

    ax.tick_params(direction='in')

    ax.set_xticks(range(len(posters)))
    ax.set_xticklabels(posters, rotation=90, fontsize=3)

    ax.set_yticks(range(len(posters)))
    ax.set_yticklabels(posters, rotation=0, fontsize=3)

    # plot the colour bar

    cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])

    cax.get_xaxis().set_visible(False)
    cax.get_yaxis().set_visible(False)

    cax.patch.set_alpha(0)
    cax.set_frame_on(False)

    plt.colorbar(orientation='vertical')
    plt.savefig(str(file_out), dpi=300)

    return


def model_q5(data, verbose):  # would you characterise any of the posters as online trolls i.e., persons attempting to inflame reactions or post off-topic messages and why?
    print('\n - Troll Analysis')

    # laod the sentiments

    file_out = Path('./results/repository/Sentiments.csv')

    if os.path.exists(file_out):
        print('\n - Loading')
        df_sentiments = pd.read_csv(file_out, encoding='iso-8859-1')

    ##############################################################################

    # merge the sentiments with tweets by the tweet ID

    df_tweets = pd.merge(left=df_sentiments, right=data, how='left', left_on='Tweet', right_on='Tweet')

    df_tweets = df_tweets[['Tweet', 'Ticker_x', 'Poster_x', 'Time_x', 'Body_x', 'Polarity', 'Subjectivity', 'Comments', 'Retweets', 'Likes']]
    df_tweets.columns = ['Tweet', 'Ticker', 'Poster', 'Time', 'Body', 'Polarity', 'Subjectivity', 'Comments', 'Retweets', 'Likes']

    # calculate the mean sentiments and interactions on monthly basis for each poster

    df_tweets['Time'] = pd.to_datetime(df_tweets['Time'])

    trolls_engages = df_tweets.groupby(['Poster', df_tweets['Time'].dt.to_period("M")]).agg({'Polarity': 'mean', 'Comments': 'sum', 'Retweets': 'sum', 'Likes': 'sum'}).reset_index(level=[0, 1])
    trolls_engages['Time'] = trolls_engages['Time'].apply(lambda x: pd.to_datetime(str(x)))
    trolls_engages['Interacts'] = trolls_engages['Comments'] + trolls_engages['Retweets'] + trolls_engages['Likes']

    trolls_engages = trolls_engages[['Poster', 'Time', 'Polarity', 'Interacts']]

    # calculate the monthly changes in sentiments and interactions

    trolls_engages['dPolarity'] = trolls_engages['Polarity'] - trolls_engages['Polarity'].shift(+1)
    trolls_engages['dInteracts'] = trolls_engages['Interacts'] - trolls_engages['Interacts'].shift(+1)

    trolls_engages = trolls_engages.sort_values(by=['Poster', 'Time'], ascending=[True, True]).reset_index(drop=True)

    trolls_engages.index += 1
    trolls_engages.index.name = 'ID'

    print('\n', tabulate(trolls_engages.head(10), headers='keys', tablefmt='simple', showindex=True, numalign='right', stralign='left'))

    ##############################################################################

    # calculate the mean of sentiment changes and its absolute value for each poster

    trolls_polarity = trolls_engages.groupby(['Poster']).agg({'dPolarity': 'mean'}).reset_index()
    trolls_polarity.columns = ['Poster', 'Polarity Diff']

    trolls_polarity['Polarity AbsDiff'] = trolls_polarity['Polarity Diff'].abs()
    trolls_polarity = trolls_polarity.sort_values(by=['Polarity AbsDiff'], ascending=False).reset_index(drop=True)

    trolls_polarity.index += 1
    trolls_polarity.index.name = 'ID'

    print('\n', tabulate(trolls_polarity, headers='keys', tablefmt='simple', showindex=True, numalign='right', stralign='left'))

    ##############################################################################

    # calculate the mean of interaction changes and its absolute value for each poster

    trolls_interacts = trolls_engages.groupby(['Poster']).agg({'dInteracts': 'mean'}).reset_index()
    trolls_interacts.columns = ['Poster', 'Interacts Diff']

    trolls_interacts['Interacts AbsDiff'] = trolls_interacts['Interacts Diff'].abs()
    trolls_interacts = trolls_interacts.sort_values(by=['Interacts AbsDiff'], ascending=False).reset_index(drop=True)

    trolls_interacts.index += 1
    trolls_interacts.index.name = 'ID'

    print('\n', tabulate(trolls_interacts, headers='keys', tablefmt='simple', showindex=True, numalign='right', stralign='left'))

    ##############################################################################
    ##############################################################################

    # load the language model

    model_path: str = str.lower('./results/repository/models/embedding.gsm')

    if os.path.exists(model_path):
        print('\n - Loading Language Model ')
        model = Doc2Vec.load(model_path)

        print('\n\t {}'.format(model).expandtabs(2))

    ##############################################################################

    # load the aggregated tweets (posts)

    file_out = Path('./results/repository/Posts.csv')

    if os.path.exists(file_out):
        print('\n - Loading')
        df_posts = pd.read_csv(file_out, encoding='iso-8859-1')

    file_out = Path('./results/repository/Outliers.csv')

    posters = list(df_posts['Poster'])
    posts = list(df_posts['Post'])

    # calculate the similarity scores among each pair of influential posters

    score = [[0 for x in range(len(posters))] for y in range(len(posters))]

    for idx1, post1 in enumerate(posts):
        for idx2, post2 in enumerate(posts):
            score[idx1][idx2] = model.n_similarity(parse(post1), parse(post2))

    trolls_similarity = pd.DataFrame()

    # calculate the mean of similarity scores for each poster

    for idx, poster in enumerate(posters):
        trolls_similarity = trolls_similarity.append({'Poster': poster, 'Aggregated Similarity': np.mean([score[idx][x] for x in range(len(posters))])}, ignore_index=True)

    # sort the posters in ascending order of the aggregated similarity to find the outliers

    trolls_similarity = trolls_similarity[['Poster', 'Aggregated Similarity']]
    trolls_similarity = trolls_similarity.sort_values(by=['Aggregated Similarity'], ascending=True).reset_index(drop=True)

    trolls_similarity.index += 1
    trolls_similarity.index.name = 'ID'

    print('\n', tabulate(trolls_similarity, headers='keys', tablefmt='simple', showindex=True, numalign='right', stralign='left'))

    # save the outliers

    trolls_similarity = trolls_similarity.applymap(lambda x: x.encode('unicode_escape').decode('iso-8859-1') if isinstance(x, str) else x)
    trolls_similarity.to_csv(str(file_out), index=False, encoding='iso-8859-1')

    return
