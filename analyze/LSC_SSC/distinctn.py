#!/usr/bin/python
# -*- coding:utf-8 -*-
import nltk


def distinctn(text, n):
    '''text: str of text
       return: unique n-gram / all n-gram'''
    tokens = nltk.word_tokenize(text)
    uni_ngram_list = []
    n_gram_num = len(tokens) - n + 1
    for i in range(n_gram_num):
        n_gram = tokens[i:i+n]
        uni_ngram_list.append(' '.join(n_gram))
    uni_ngram_list = set(uni_ngram_list)
    if n_gram_num == 0:
        return 1
    return len(uni_ngram_list) / n_gram_num


if __name__ == '__main__':
    examples = [
                'A short story is a piece of prose fiction that typically can be read in one sitting and focuses on a self-contained incident or series of linked incidents, with the intent of evoking a single effect or mood. The short story is one of the oldest types of literature and has existed in the form of legends, mythic tales, folk tales, fairy tales, fables and anecdotes in various ancient communities across the world. The modern short story developed in the early 19th century.',
                'Some authors have argued that a short story must have a strict form. Somerset Maugham thought that the short story "must have a definite design, which includes a point of departure, a climax and a point of test; in other words, it must have a plot". This view is however opposed by Anton Chekov who thought that a story should have neither a beginning nor an end. It should just be a "slice of life", presented suggestively.',
                "Short stories have no set length. In terms of word count, there is no official demarcation between an anecdote, a short story, and a novel. Rather, the form's parameters are given by the rhetorical and practical context in which a given story is produced and considered so that what constitutes a short story may differ between genres, countries, eras, and commentators. Like the novel, the short story's predominant shape reflects the demands of the available markets for publication, and the evolution of the form seems closely tied to the evolution of the publishing industry and the submission guidelines of its constituent houses."
            ]
    for eg in examples:
        print(eg)
        for i in [2, 3, 4]:
            print(f'distinct-{i}: {distinctn(eg, i)}')
