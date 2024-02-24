# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 19:15:58 2024

@author: jaege
"""

from transformers import GPT2LMHeadModel, GPT2Tokenizer

# GPT-2 모델 및 토크나이저 로드
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# 문장 생성 함수
def generate_sentence(prompt, max_length=50):
    input_ids = tokenizer.encode(prompt, return_tensors="pt", max_length=max_length, truncation=True)
    output_ids = model.generate(input_ids, max_length=max_length, 
                                num_return_sequences=1, 
                                pad_token_id = 100,
                                no_repeat_ngram_size=2)
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated_text

# 문장 생성 예제
prompt = "In a galaxy far, far away"
#prompt = "Seoul is the one of the best cities to live."
#prompt = "The earth is a unique planet to live human being"

generated_sentence = generate_sentence(prompt, max_length=100)
print(generated_sentence)