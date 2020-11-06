# from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def extract(string, use='vader'):
    if string is None or string == '':
        return -1

    # if use == 'blob':
    #    blob = TextBlob(string)
    #    polarity, subjectivity = blob.sentiment
    #    return (polarity + 1) / 2

    if use == 'vader':
        # Create a SentimentIntensityAnalyzer object.
        sid_obj = SentimentIntensityAnalyzer()

        # polarity_scores method of SentimentIntensityAnalyzer
        # object gives a sentiment dictionary.
        # which contains pos, neg, neu, and compound scores.
        sentiment_dict = sid_obj.polarity_scores(string)

        # print("Overall sentiment dictionary is : ", sentiment_dict)
        # print("sentence was rated as ", sentiment_dict['neg'] * 100, "% Negative")
        # print("sentence was rated as ", sentiment_dict['neu'] * 100, "% Neutral")
        # print("sentence was rated as ", sentiment_dict['pos'] * 100, "% Positive")

        return (sentiment_dict['compound'] + 1) / 2

    return -1


if __name__ == '__main__':
    # print(extract(input()))
    phrases = ['fantastical reality',
        'melodramatic',
        'mellow',
        'Like the feeling after you take a warm shower and get into bed after a long day',
        'rock',
        'relaxing',
        'happy',
        'chill love songs',
        'cheerful and upbeat',
        'catchy',
        'looking out at the ocean on a deserted beach',
        'a rainy night',
        'Something that can help me exercise',
        'a sweet dream',
        'emotional love songs',
        'Something fun',
        'Blazing Hot Fire',
        'Girly',
        'Birthday party',
        'Wild and intense',
        'Serene and Tranquil',
        'inspiring and grand',
        'Happy',
        'Something I can relax and chill to']

    for phrase in phrases:
        print('{}: {}'.format(phrase, extract(phrase.lower()) * 10))

