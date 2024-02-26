# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 22:06:26 2024

@author: jaege
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data.metrics import bleu_score
from torch.utils.tensorboard import SummaryWriter
from torchtext.data import Field, BucketIterator
import torchtext
from konlpy.tag import Okt
import spacy
import sys, time

# download SpaCy models and loadings
#spacy.cli.download("en_core_web_sm")
#spacy.cli.download("ko_core_news_sm")

spacy_en = spacy.load("en_core_web_sm")
spacy_ge = spacy.load("ko_core_news_sm")

# define tokenize
def tokenize_eng(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]
def tokenize_ko(text):
    okt = Okt()
    tokens = okt.morphs(text)
    return tokens

# Define Fields
korean = Field(tokenize=tokenize_ko, lower=True, init_token="<sos>", eos_token="<eos>")
english = Field(tokenize=tokenize_eng, lower=True, init_token="<sos>", eos_token="<eos>")

# Load Datasets and splits into 3 data sets, the datasets must be in folder 'path'
# named correctly
train_data, valid_data, test_data = torchtext.data.TabularDataset.splits(
    path="/Users/jaege/TestPGM/NLP", train="train_data.csv", 
    validation="valid_data.csv", test="test_data.csv", 
    format="csv",
    fields=[("src", english), ("trg", korean)]
)

# build vocabularies
english.build_vocab(train_data, max_size= 1000, min_freq=2)
korean.build_vocab(train_data,  max_size= 1000, min_freq=2)

# Transformer define
class Transformer(nn.Module):
    def __init__(
        self,
        embedding_size,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        forward_expansion,
        dropout,
        max_len,
        device,
    ):
        super(Transformer, self).__init__()
        self.src_word_embedding = nn.Embedding(src_vocab_size, embedding_size)
        self.src_position_embedding = nn.Embedding(max_len, embedding_size)
        self.trg_word_embedding = nn.Embedding(trg_vocab_size, embedding_size)
        self.trg_position_embedding = nn.Embedding(max_len, embedding_size)

        self.device = device
        self.transformer = nn.Transformer(
            embedding_size,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            forward_expansion,
            dropout,
        )
        self.fc_out = nn.Linear(embedding_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.src_pad_idx = src_pad_idx

    def make_src_mask(self, src):
        src_mask = src.transpose(0, 1) == self.src_pad_idx

        # (N, src_len)
        return src_mask.to(self.device)

    def forward(self, src, trg):
        src_seq_length, N = src.shape
        trg_seq_length, N = trg.shape

        src_positions = (
            torch.arange(0, src_seq_length)
            .unsqueeze(1)
            .expand(src_seq_length, N)
            .to(self.device)
        )

        trg_positions = (
            torch.arange(0, trg_seq_length)
            .unsqueeze(1)
            .expand(trg_seq_length, N)
            .to(self.device)
        )

        embed_src = self.dropout(
            (self.src_word_embedding(src) + self.src_position_embedding(src_positions))
        )

        embed_trg = self.dropout(
            (self.trg_word_embedding(trg) + self.trg_position_embedding(trg_positions))
        )

        src_padding_mask = self.make_src_mask(src)
        trg_mask = self.transformer.generate_square_subsequent_mask(trg_seq_length).to(
            self.device
        )

        out = self.transformer(
            embed_src,
            embed_trg, # added [:-1,:]
            src_key_padding_mask=src_padding_mask,
            tgt_mask=trg_mask,
        )
        out = self.fc_out(out)
        return out

# Define traslate sentence 
def translate_sentence(model, sentence, korean, english, device, max_length=50):
    # Create tokens using spacy and everything in lower case (which is what our vocab is)
    if type(sentence) == str:
        tokens = [korean.init_token] + [token.lower() for token in korean.tokenize(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    # Add <SOS> and <EOS> in beginning and end respectively
    tokens.insert(0, korean.init_token)
    tokens.append(korean.eos_token)

    # Go through each korean token and convert to an index
    text_to_indices = [korean.vocab.stoi[token] for token in tokens]

    # Convert to Tensor
    sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)

    outputs = [english.vocab.stoi["<sos>"]]
    losses = []
    
    for i in range(max_length):
        trg_tensor = torch.LongTensor(outputs).unsqueeze(1).to(device)

        with torch.no_grad():
            output = model(sentence_tensor, trg_tensor)

#        print("trg, output shape", trg_tensor.shape, output.shape)
        
        best_guess = output.argmax(2)[-1, :].item()
        outputs.append(best_guess)

        if best_guess == english.vocab.stoi["<eos>"]:
            break
    
    translated_sentence = [english.vocab.itos[idx] for idx in outputs]
    # remove start token
    return translated_sentence[1:]

def bleu(data, model, english, korean, device):
    targets = []
    outputs = []

    for example in data:
        src = vars(example)["src"]
        trg = vars(example)["trg"]

        prediction = translate_sentence(model, src, english, korean, device)
        prediction = prediction[:-1]  # remove <eos> token

        targets.append([trg])
        outputs.append(prediction)

    return bleu_score(outputs, targets)

# load saved training model for prediction without training
def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    
# We're ready to define everything we need for training our Seq2Seq model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training hyperparameters
learning_rate = 3e-4
batch_size = 64

# Model hyperparameters
src_vocab_size = len(english.vocab)
trg_vocab_size = len(korean.vocab)
embedding_size = 512
num_heads = 8
num_encoder_layers = 4
num_decoder_layers = 4
dropout = 0.10
max_len = 256
forward_expansion = 4
src_pad_idx = english.vocab.stoi["<pad>"]

step = 0

# Build Transformer for training
model = Transformer(
    embedding_size,
    src_vocab_size,
    trg_vocab_size,
    src_pad_idx,
    num_heads,
    num_encoder_layers,
    num_decoder_layers,
    forward_expansion,
    dropout,
    max_len,
    device,
).to(device)

#model.load_state_dict(torch.load("/Users/jaege/TestPGM/NLP/seq2seqCheck"))  # 모델 경로를 적절히 지정
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
load_checkpoint(torch.load("/Users/jaege/TestPGM/NLP/seq2seqCheck250"), model, optimizer)
model.to(device)

# sentence to be translate to Korean
while True :
    sentence = input("영어 문장을 입력하세요 : ")

    model.eval()
    translated_sentence = translate_sentence(
            model, sentence, english, korean, device, max_length=50
            )
    print(f"Translated example sentence:  {translated_sentence}")
    if sentence == "q":
        break
