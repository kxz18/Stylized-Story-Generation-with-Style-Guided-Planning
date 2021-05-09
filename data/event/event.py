import nltk
import random
import pickle
import argparse
import pandas
from scipy.stats import norm
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


stemmer = WordNetLemmatizer()
trash_words = ['[', ']', 'MALE', 'FEMALE', 'NEUTRAL']

mean = None
sigma = None
discard_words = None


def judge_event(text):
    global stemmer, discard_words
    global mean, sigma
    if mean is None or sigma is None:
        path = __file__.replace('event.py', 'stats.data')
        with open(path, 'rb') as fin:
            mean, sigma = pickle.load(fin)

    if discard_words is None:
        prefix = __file__.replace('event.py', '')
        with open(prefix + 'idf_cache.pk', 'rb') as f:
            idf = pickle.load(f)
        with open(prefix + 'tf_idf_cache.pk', 'rb') as f:
            tf_idf = pickle.load(f)
        with open(prefix + 'event_word_cache.pk', 'rb') as f:
            event_word = pickle.load(f)
        event_word.sort(key=lambda x:idf[x]) # ascending order
        discard_words = event_word[:10]

    event_word = []
    tokens = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(tokens)
    for word, tags in tagged:
        if ('[' in word) or (']' in word) or(word in trash_words):
            continue
        if 'VB' in tags:
            origin_word = stemmer.lemmatize(word, 'v')
            if origin_word in stopwords.words('english'):
                continue
            if origin_word in discard_words:
                continue
            event_word.append(word)
    event_word = list(set(event_word))
    val = (len(event_word) - mean) / sigma
    return norm.cdf(val)


if __name__ == '__main__':
    print(judge_event("Oh my god, this is too hard"))
