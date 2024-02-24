# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 19:14:09 2024

@author: jaege
"""

import random

# 단어들의 리스트
subjects = ["I", "You", "He", "She", "We", "They"]
verbs = ["eat", "sleep", "run", "study", "write"]
objects = ["an apple", "a book", "the beach", "music", "a movie"]

# 무작위로 문장 생성
def generate_sentence():
    subject = random.choice(subjects)
    verb = random.choice(verbs)
    obj = random.choice(objects)

    sentence = f"{subject} {verb} {obj}."
    return sentence

# 여러 문장 생성
for _ in range(5):
    print(generate_sentence())