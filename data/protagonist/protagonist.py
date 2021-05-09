#!/usr/bin/python
# -*- coding:utf-8 -*-
import nltk


def extract_protagonist(text):
    '''extract protagonist from given text'''
    tagged = nltk.pos_tag(nltk.word_tokenize(text))
    entities = []

    for tree in nltk.ne_chunk(tagged, binary=True).subtrees():
        # filter the root
        if tree.label() == "S":
            continue
        entity_name = ' '.join(c[0] for c in tree.leaves())
        entities.append(entity_name)
    if len(entities) == 0:
        # search through nouns
        for word in filter(lambda x: x[1] == 'NN' or x[1] == 'PRP' or x[1] == 'RB', tagged):
            if word[1] == 'NN':
                entities.append(word[0])
            elif word[1] == 'PRP' and (word[0] == 'I' or word[0] == 'We'):
                entities.append(word[0])
            elif word[1] == 'RB' and word[0][0].isupper():  # Sally, Eliza
                entities.append(word[0])
            else:
                continue
            break
    special_cases = ['girls', 'kids', 'bugs']
    for case in special_cases:
        if case in text:
            entities.append(case)
    entities.sort(key=lambda x: text.count(x), reverse=True)
    if len(entities) == 0:  # just cannot find propriate protagonist
        return None
    return entities[0]


if __name__ == '__main__':
    examples = ['I am a student. John is too. That dog looks good. Sally is good.']
    for eg in examples:
        print(f'Story: {eg}')
        print(extract_protagonist(eg))
