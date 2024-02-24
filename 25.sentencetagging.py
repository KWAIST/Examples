# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 19:12:10 2024

@author: jaege
"""

import nltk
from nltk import sent_tokenize, word_tokenize, pos_tag, RegexpParser

# nltk 데이터 다운로드 (한 번만 실행하면 됨)
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# 예제 문장
example_sentence = "Natural language processing is a subfield of artificial intelligence."

# 문장을 단어로 토큰화
words = word_tokenize(example_sentence)

# 단어에 품사 태깅
tagged_words = pos_tag(words)

# 정규 표현식을 사용한 구문 분석
grammar = "NP: {<DT>?<JJ>*<NN>}"
chunk_parser = RegexpParser(grammar)
tree = chunk_parser.parse(tagged_words)

# 결과 출력
print("토큰화된 문장:", words)
print("품사 태깅 결과:", tagged_words)
print("\n구문 분석 결과:")
tree.pretty_print()