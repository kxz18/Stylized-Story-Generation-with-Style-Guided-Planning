#!/usr/bin/python
# -*- coding:utf-8 -*-
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
import nltk.sentiment.util as util
import nltk.sentiment.sentiment_analyzer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from scipy.stats import norm
from nrclex import NRCLex


sid = SentimentIntensityAnalyzer()
examples = ["I'm happy to know that you are depressed by him"]
stop = stopwords.words('english') + list(string.punctuation) + ['FEMALE', 'MALE', 'NEUTRAL']
mean = None
sigma = None
lemmatizer = WordNetLemmatizer()
golden = ['thrilled', 'like', 'liked', 'love', 'loved']


def penn_to_wn(tag):
    '''convert pos tag to simple wordnet tags'''
    mapping = {
            'J': wn.ADJ,
            'N': wn.NOUN,
            'R': wn.ADV,
            'V': wn.VERB
            }
    for start in mapping:
        if tag.startswith(start):
            return mapping[start]
    return None


def get_senti(word, tag):
    '''get sentiment score of a single word'''
    word = word.lower()
    wn_tag = penn_to_wn(tag)

    if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV):
        return []


    synsets = wn.synsets(word, pos=wn_tag)
    if not synsets:
        return []

    # Take the first sense, the most common
    synset = synsets[0]
    swn_synset = swn.senti_synset(synset.name())
    return [swn_synset.pos_score(),swn_synset.neg_score(),swn_synset.obj_score()]


def get_emotional_words(text, pos=False):
    text_object = NRCLex(text)
    affect_dict = text_object.affect_dict
    emotional_words_set = []
    no_use_class = set(['negative', 'positive', 'anticipation', 'trust'])
    for word in affect_dict:
        emo_mapping = set(affect_dict[word]).difference(no_use_class)
        if len(emo_mapping) == 0:
            continue
        # has been mapped to real emotion instead of just positive and negative
        emotional_words_set.append(word)
    keywords_list = []
    pos_list = []
    for sent_cnt, sent in enumerate(text_object.sentences):
        for word in sent.words:
            if word in emotional_words_set or word in golden:
                keywords_list.append(word)
                pos_list.append(sent_cnt)
    if pos:
        return keywords_list, pos_list
    return keywords_list
    # ref1 = list(text_object.affect_dict.keys())
    # print(text_object.affect_dict)
    # global sid, stop, golden

    # ref2 = []
    # ref3 = []
    # words = [w for w in nltk.word_tokenize(text) if w not in stop]
    # tagged = nltk.pos_tag(words)
    # for word, tag in tagged:
    #     try:
    #         # to original form
    #         wn_tag = penn_to_wn(tag)
    #         lemma_word = lemmatizer.lemmatize(word, pos=wn_tag)
    #     except KeyError:
    #         lemma_word = word
    #     # method 2
    #     scores = get_senti(lemma_word, tag)
    #     if len(scores) == 0:
    #         continue
    #     if scores[0] > 0 or scores[1] > 0:  # pos score and neg score
    #         ref2.append(word)
    #     # method 3
    #     ss = sid.polarity_scores(lemma_word)
    #     for key in ss:
    #         if ss[key] > 0.5 and (key == 'pos' or key == 'neg'):
    #             ref3.append(word)
    # ref1 = set(ref1)
    # ref2 = set(ref2)
    # ref3 = set(ref3)
    # inter1 = ref1.intersection(ref2)
    # inter2 = ref2.intersection(ref3)
    # inter3 = ref3.intersection(ref1)
    # emotional_words_set = inter1.union(inter2).union(inter3)
    # keywords_list = []
    # for word in words:
    #     if word in emotional_words_set or word in golden:
    #         keywords_list.append(word)
    # return keywords_list


def judge_emotional(text):
    global mean, sigma
    if mean is None or sigma is None:
        path = __file__.replace('emotional.py', 'stats.data')
        with open(path, 'rb') as fin:
            mean, sigma = pickle.load(fin)
    keywords = get_emotional_words(text)
    val = (len(keywords) - mean) / sigma
    return norm.cdf(val)


if __name__ == '__main__':
    print(__file__)
    for text in examples:
        print(text)
        print(get_emotional_words(text, pos=True))
        print(judge_emotional(text))
