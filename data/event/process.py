import nltk
import random
import pickle
import argparse
import pandas
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

'''
create dataset by results of create_tf_idf.py
'''

data_dir = './ROCStories_Masked.csv'
stemmer = WordNetLemmatizer()
trash_words = ['[', ']', 'MALE', 'FEMALE']
total_len = 0

def gen_event_for_one_line(line, metric, num=5, discard_words=[], DEBUG=False):
    global stemmer
    global total_len

    if line.name % 500 == 0:
        print(line.name, end='\r')
    columns = [f'sentence{i}' for i in range(2,6)]
    event_word = []
    for col_name in columns:
        sent = line[col_name]
        print('sent = \n'+sent)
        tokens = nltk.word_tokenize(sent)
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
                # word_metric = metric[stemmer.lemmatize(word)]
                event_word.append(word)
    event_word = list(set(event_word))
    event_word.sort(key=lambda x:metric[stemmer.lemmatize(x, 'v')], reverse=True) # descending order
    total_len += len(event_word)
    return ','.join(event_word)    # TODO only use the first `num` words

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('num', action='store', default=10, type=int, help='num to discard, ex: 10 means to discard the last 10\
         event words in tf_idf')
    parser.add_argument('load', action='store', type=str, help='path to save the tf_idf and event word')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse()
    DEBUG = args.debug

    # read data
    df = pandas.read_csv(data_dir)
    if DEBUG:
        # rand_start = random.randint(0, 60000)
        rand_start = 566
        print(f'story start from {rand_start} to {rand_start + 10}')
        df = df.iloc[rand_start:rand_start+10]
    number = df.shape[0]
    print(f'all = {number}')

    with open('idf_'+args.load, 'rb') as f:
        idf = pickle.load(f)
    with open('tf_idf_'+args.load, 'rb') as f:
        tf_idf = pickle.load(f)
    with open('event_word_'+args.load, 'rb') as f:
        event_word = pickle.load(f)
    if DEBUG:
        event_word.sort(key=lambda x:idf[x]) # ascending order
        print(f'{args.num} Minimum idf words : {[(word, idf[word]) for word in event_word[:args.num]]}')
        event_word.sort(key=lambda x:tf_idf[x]) # ascending order
        print(f'{args.num} Minimum tf_idf words : {event_word[:args.num]}')
    # filter by some strategy
    event_word.sort(key=lambda x:idf[x]) # ascending order
    event = df.apply(gen_event_for_one_line, axis=1, args=(idf, args.num, event_word[:10]))
    df['event'] = event
    df.to_csv('./processed_ROC.csv', index=False)
    print(f'avg len = {total_len/number}')
