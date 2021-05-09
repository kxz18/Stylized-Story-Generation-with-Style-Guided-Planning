import nltk
from nltk.text import TextCollection
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas
import argparse
import pickle
import random

data_dir = './ROCStories_Masked.csv'
stemmer = WordNetLemmatizer()
trash_words = ['[', ']', 'MALE', 'FEMALE']


def gen_event_for_one_line(line, DEBUG=False):
    global stemmer
    global trash_words

    columns = [f'sentence{i}' for i in range(2,6)]
    event_word = []
    for col_name in columns:
        sent = line[col_name]
        tokens = nltk.word_tokenize(sent)
        # print(' '.join(tokens), ' '.join(stemmed), sep='\n')
        tagged = nltk.pos_tag(tokens)
        for word, tags in tagged:
            if word in trash_words:
                continue
            if 'VB' in tags:
                word = stemmer.lemmatize(word, 'v') # transfer to stem
                if word in stopwords.words('english'):
                    continue
                event_word.append(word)
    return list(set(event_word))

def tokenize_for_one_line(line, DEBUG=False):
    columns = [f'sentence{i}' for i in range(2,6)]
    tokens = []
    for col in columns:
        sent = line[col]
        tokens += nltk.word_tokenize(sent)
    return tokens


def compute_tf_idf(df, DEBUG=False):
    tf_idf = {}
    idf = {}
    story_event_word = []   # each elements is a list
    event = df['event'].values.tolist()
    for sent_event_word in event:
        story_event_word.append(sent_event_word.split(', '))
    word_collection = TextCollection(story_event_word)
    all_word_li = [w for text in story_event_word for w in text]    # all words without deduplication
    if DEBUG:
        print(f'event words for each text\n{story_event_word}\n')
        print(f'total event words {all_word_li}')
    event_word = set(all_word_li)
    total_word_num = len(event_word)

    # calculate idf, tf_idf
    for i, word in enumerate(event_word):
        if i % 500 == 0:
            print(f'[calculate idf and tf_idf] : {i + 1}/{total_word_num}\r', end='')
        idf[word] = word_collection.idf(word)
        tf_idf[word] = word_collection.tf_idf(word, all_word_li)
    print('')
    return idf, tf_idf, event_word

def create(df, DEBUG=False):
    df['event'] = None
    totol_line = df.shape[0]
    for index in df.index:
        if DEBUG:
            print(index)
        if index % 500 == 0:
            print(f'processed {index} / {totol_line}\r', end='')
        line = df.loc[index]
        words = gen_event_for_one_line(line)
        if 'be' in words:
            print(line) # print if be in it
        df.loc[index, 'event'] = ', '.join(list(words))
    print('')
    return compute_tf_idf(df, DEBUG)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('num', action='store', default=10, type=int, help='num to discard, ex: 10 means to discard the last 10\
         event words in tf_idf')
    parser.add_argument('save', action='store', type=str, help='path to save the tf_idf and event word')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    args = parser.parse_args()
    DEBUG = args.debug

    df = pandas.read_csv(data_dir)
    if DEBUG:
        rand_start = random.randint(0, 10000)
        print(f'story start from {rand_start} to {rand_start + 5}')
        df = df.iloc[rand_start:rand_start+5]

    idf, tf_idf, event_word = create(df, DEBUG)
    event_word = list(event_word)
    if DEBUG:
        print('event word :', f'{event_word}\n', sep='\n')
    if args.save:
        with open('idf_'+args.save, 'wb') as f:
            pickle.dump(idf, f)
        with open('tf_idf_'+args.save, 'wb') as f:
            pickle.dump(tf_idf, f)
        with open('event_word_'+args.save, 'wb') as f:
            pickle.dump(event_word, f)

    print(f'event_word length = {len(event_word)}\n')
    event_word.sort(key=lambda x:idf[x])   # sort by ascending order of idf
    print(f'{args.num} Minimum idf words : {event_word[:args.num]}\n')

    event_word.sort(key=lambda x:tf_idf[x])
    print(f'{args.num} Minimum tf_idf words : {event_word[:args.num]}\n')
    if DEBUG:
        for k in event_word:
            print(f'[{k}]: {idf[k]}')
    # override the previous event column
    # event = df.apply(gen_event_for_one_line, axis=1, args=(event_word[:args.num], ))
    # df['event'] = event

    # df.to_csv('./processed_ROC.csv')
    print('complete')
