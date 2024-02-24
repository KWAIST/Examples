# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 19:08:33 2024

@author: jaege
"""

import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# WordNetLemmatizer 초기화
lemmatizer = WordNetLemmatizer()

# WordNet 품사 태그를 NLTK 품사 태그로 매핑
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return 'a'  # 형용사
    elif tag.startswith('V'):
        return 'v'  # 동사
    elif tag.startswith('R'):
        return 'r'  # 부사
    else:
        return 'n'  # 명사

# 예제 문장
example_sentence = "He is running and eating at the same time."

# 문장을 단어로 토큰화
words = word_tokenize(example_sentence)

# 품사 태깅
pos_tags = nltk.pos_tag(words)

# 표제어 추출
lemmatized_words = [lemmatizer.lemmatize(word, pos=get_wordnet_pos(pos)) for word, pos in pos_tags]

# 결과 출력
print("토큰화된 문장:", words)
print("품사 태깅 결과:", pos_tags)
print("표제어 추출 결과:", lemmatized_words)