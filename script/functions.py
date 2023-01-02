import pandas as pd
import numpy as np
import re
import snscrape.modules.twitter as sntwitter
from transformers import pipeline
import plotly.express as px
from sentence_transformers import SentenceTransformer



def get_tweets(username, length=10, option = None):
    # Creating list to append tweet data to
    query = "("+username + ")"+"(to:"+username+") -filter:links filter:replies"
    if option == "Advanced":
        query = username
    tweets = []
    # Using TwitterSearchScraper to scrape
    # Using TwitterSearchScraper to scrape
    for i,tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
        if i>=length:
            break
        tweets.append([tweet.content])
    
    # Creating a dataframe from the tweets list above
    tweets_df = pd.DataFrame(tweets, columns=["content"])
    tweets_df['content'] = tweets_df['content'].str.replace('@[^\s]+','')
    tweets_df['content'] = tweets_df['content'].str.replace('#[^\s]+','')
    tweets_df['content'] = tweets_df['content'].str.replace('http\S+','')
    tweets_df['content'] = tweets_df['content'].str.replace('pic.twitter.com\S+','')
    tweets_df['content'] = tweets_df['content'].str.replace('RT','')
    tweets_df['content'] = tweets_df['content'].str.replace('amp','')
    # remove emoticon
    tweets_df['content'] = tweets_df['content'].str.replace('[^\w\s#@/:%.,_-]', '', flags=re.UNICODE)

    # remove whitespace leading & trailing
    tweets_df['content'] = tweets_df['content'].str.strip()

    # remove multiple whitespace into single whitespace
    tweets_df['content'] = tweets_df['content'].str.replace('\s+', ' ')

    # remove row with empty content
    tweets_df = tweets_df[tweets_df['content'] != '']
    return tweets_df


def get_sentiment(df):
    # Sentiment Analysis
    classifier = pipeline("sentiment-analysis",model = "indobert")
    df['sentiment'] = df['content'].apply(lambda x: classifier(x)[0]['label'])
    # change order sentiment to first column
    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df = df[cols]

    return df

def get_bar_chart(df):
    df= df.groupby(['sentiment']).count().reset_index()
    # plot barchart sentiment
   # plot barchart sentiment
    fig = px.bar(df, x="sentiment", y="content", color="sentiment",text = "content", color_discrete_map={"positif": "#00cc96", "negatif": "#ef553b","netral": "#636efa"})
    # hide legend
    fig.update_layout(showlegend=False)
    # set margin top 
    fig.update_layout(margin=dict(t=0, b=100, l=0, r=0))
    # set title in center
    # set annotation in bar
    fig.update_traces(textposition='outside')
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

    # set y axis title
    fig.update_yaxes(title_text='Jumlah Komentar')

    return fig
