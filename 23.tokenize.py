# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 19:05:56 2024

@author: jaege
"""

from konlpy.tag import Okt  # KoNLPy의 Twitter 형태소 분석기를 사용

def tokenize(text):
    okt = Okt()
    tokens = okt.pos(text)
    return tokens

# 예제 문장
example_sentence = "자연어 처리는 인공지능의 중요한 분야 중 하나입니다."

# 형태소 분석 실행
result = tokenize(example_sentence)

# 결과 출력
for token in result:
    print(f"형태소: {token[0]}, 품사: {token[1]}")