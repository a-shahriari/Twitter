# Twitter vs Stock Market

The Tweets about the Top Companies from 2015 to 2020 dataset was created to determine possible speculators and influencers in a stock market. It was published as a paper in the 2020 IEEE International Conference on Big Data. It includes tweets about top technology firms in US uploaded to Kaggle platform in accordance with the CC0 Public Domain licencing agreement.

The dataset contains over 3 million unique tweets with their information such as tweet id, author of the tweet, postdate, the text body of the tweet, and the number of comments, likes, and retweets of tweets matched with the related company. These tweets are collected from Twitter by a parsing script that is based on Selenium. The inspiration behind this collection is conducting research on determining the correlation between the market value of company respect to the public opinion of that company, sentiment analysis of the companies with a time series, and evaluating troll users who try to occupy the social agenda.

To search for the speculators and influencers in the stock market, we use a dataset called Values of Top NASDAQ Companies from 2010 to 2020. This dataset includes the daily market share values changes of Amazon (AMZN), Apple (APPL), Google (GOOG, GOOGL) , Microsoft (MSFT), and Tesla (TSLA) from mid-2010 to mid-2020. It contains daily open, close, high, and low values and volume tagged by dates fetched from the official NASDAQ website. From now on, we use the above share-ticker symbols accordingly.

This analysis was performed on a HP Pavilion laptop (Intel® Core™ i7-8550U CPU @ 1.80GHz 4 Cores equipped with 32GB of dynamic RAM) employing Anaconda 1.10.0 (Individual Distribution) for Windows and PyCharm 2019.3.5 (Community Edition) as open-source development platforms. Pyhton 3.7 is the programming language of choice because of its fast-prototyping capability and wide spectrum of supporting data science libraries.

Due to upload limitations (25MB) by GitHub, the tweets and some of the results could not be uploaded but they can be provided with other means upon request.
