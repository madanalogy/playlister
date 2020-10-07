from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def extract(string, use='vader'):
    if string is None or string == '':
        return -1

    if use == 'blob':
        blob = TextBlob(string)
        polarity, subjectivity = blob.sentiment
        return (polarity + 1) / 2

    if use == 'vader':
        # Create a SentimentIntensityAnalyzer object.
        sid_obj = SentimentIntensityAnalyzer()

        # polarity_scores method of SentimentIntensityAnalyzer
        # oject gives a sentiment dictionary.
        # which contains pos, neg, neu, and compound scores.
        sentiment_dict = sid_obj.polarity_scores(string)

        # print("Overall sentiment dictionary is : ", sentiment_dict)
        # print("sentence was rated as ", sentiment_dict['neg'] * 100, "% Negative")
        # print("sentence was rated as ", sentiment_dict['neu'] * 100, "% Neutral")
        # print("sentence was rated as ", sentiment_dict['pos'] * 100, "% Positive")

        return (sentiment_dict['compound'] + 1) / 2

    return -1


if __name__ == '__main__':
    print(extract(input()))
