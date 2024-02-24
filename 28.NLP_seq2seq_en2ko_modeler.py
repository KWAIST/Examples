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
import time

# download SpaCy models and loadings
spacy.cli.download("en_core_web_sm")
spacy.cli.download("ko_core_news_sm")

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
    for i in range(max_length):
        trg_tensor = torch.LongTensor(outputs).unsqueeze(1).to(device)

        with torch.no_grad():
            output = model(sentence_tensor, trg_tensor)

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

# trained model saving for future using without training step
def save_checkpoint(state, filename="/Users/jaege/TestPGM/NLP/seq2seqCheck"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

# load saved training model for prediction without training
def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

# We're ready to define everything we need for training our Seq2Seq model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

load_model = False # change it to True for loading saved trained model
save_model = True

# Training hyperparameters
num_epochs = 1000
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

# Tensorboard to get nice loss plot
writer = SummaryWriter("/Users/jaege/TestPGM/NLP")
step = 0

# split dataset for training by batch_size
train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=batch_size,
    sort_within_batch=True,
    sort_key=lambda x: len(x.src),
    device=device,
)

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

#Optimizer define, lr is learning_rate define above as hyper parameters
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.1, patience=10, verbose=True
)

pad_idx = english.vocab.stoi["<pad>"]
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

if load_model:
    load_checkpoint(torch.load("/Users/jaege/TestPGM/NLP/seq2seqCheck"), model, optimizer)

# Start Training
for epoch in range(num_epochs):
    
    # Start Training
    model.train()
    losses = []

    for batch_idx, batch in enumerate(train_iterator):
        # Get input and targets and get to cuda
        inp_data = batch.src.to(device)
        target = batch.trg.to(device)

        # Forward prop
        output = model(inp_data, target[:-1, :])

        # Output is of shape (trg_len, batch_size, output_dim) but Cross Entropy Loss
        # doesn't take input in that form. For example if we have MNIST we want to have
        # output to be: (N, 10) and targets just (N). Here we can view it in a similar
        # way that we have output_words * batch_size that we want to send in into
        # our cost function, so we need to do some reshapin.
        # Let's also remove the start token while we're at it
        output = output.reshape(-1, output.shape[2])
        target = target[1:].reshape(-1)

        optimizer.zero_grad()

        loss = criterion(output, target)
        losses.append(loss.item())

        # Back prop
        loss.backward()
        # Clip to avoid exploding gradient issues, makes sure grads are
        # within a healthy range
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

        # Gradient descent step
        optimizer.step()

        # plot to tensorboard
        writer.add_scalar("Training loss", loss, global_step=step)
        step += 1

    mean_loss = sum(losses) / len(losses)
    scheduler.step(mean_loss)
    print(f"[Epoch {epoch} / {num_epochs}]", "loss = ", mean_loss, time.strftime('= %H:%M:%S'))
    
    if (epoch % 100) == 0: # set model saving and testingfrequecy
        if save_model:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                }
            save_checkpoint(checkpoint)

        # sentence to be translate to Korean
        sentence = "tom wants peace."
        
        # Display progress of translation during each training epochs.
        model.eval()
        translated_sentence = translate_sentence(
            model, sentence, english, korean, device, max_length=50
            )
        print(f"Translated example sentence:  {translated_sentence}")

# running on entire test data takes a while
score = bleu(test_data[1:100], model, english, korean, device)
print(f"Bleu score {score * 100:.2f}")
