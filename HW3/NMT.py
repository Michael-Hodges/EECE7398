#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import io
import argparse

import numpy as np
import spacy

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data

from torchtext.datasets import TranslationDataset
from torchtext.data import Field, BucketIterator
from torchtext.vocab import Vocab, Vectors

import time
import math
import random

spacy_en = spacy.load('en')
spacy_vi = spacy.load('vi_spacy_model')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SRC = Field(tokenize = None,
			# tokenizer_language="en",
			init_token = '<sos>',
			eos_token = '<eos>',
			lower = True)

TRG = Field(tokenize = None,
			# tokenizer_language="vi_spacy_model",
			init_token = '<sos>',
			eos_token = '<eos>',
			lower = True)

class Encoder(nn.Module):
	def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
		super().__init__()
		
		self.hid_dim = hid_dim
		self.n_layers = n_layers
		
		self.embedding = nn.Embedding(input_dim, emb_dim)
		
		self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout)
		
		self.dropout = nn.Dropout(dropout)
		
	def forward(self, src):
		
		#src = [src len, batch size]
		print("source size: {}".format(src.shape))
		embedded = self.dropout(self.embedding(src))
		
		#embedded = [src len, batch size, emb dim]
		
		outputs, (hidden, cell) = self.rnn(embedded)
		
		#outputs = [src len, batch size, hid dim * n directions]
		#hidden = [n layers * n directions, batch size, hid dim]
		#cell = [n layers * n directions, batch size, hid dim]
		
		#outputs are always from the top hidden layer
		
		return hidden, cell
class Decoder(nn.Module):
	def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
		super().__init__()
		
		self.output_dim = output_dim
		self.hid_dim = hid_dim
		self.n_layers = n_layers
		
		self.embedding = nn.Embedding(output_dim, emb_dim)
		
		self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout)
		
		self.fc_out = nn.Linear(hid_dim, output_dim)
		
		self.dropout = nn.Dropout(dropout)
		
	def forward(self, input, hidden, cell):
		
		#input = [batch size]
		#hidden = [n layers * n directions, batch size, hid dim]
		#cell = [n layers * n directions, batch size, hid dim]
		
		#n directions in the decoder will both always be 1, therefore:
		#hidden = [n layers, batch size, hid dim]
		#context = [n layers, batch size, hid dim]
		
		input = input.unsqueeze(0)
		
		#input = [1, batch size]
		
		embedded = self.dropout(self.embedding(input))
		
		#embedded = [1, batch size, emb dim]
				
		output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
		
		#output = [seq len, batch size, hid dim * n directions]
		#hidden = [n layers * n directions, batch size, hid dim]
		#cell = [n layers * n directions, batch size, hid dim]
		
		#seq len and n directions will always be 1 in the decoder, therefore:
		#output = [1, batch size, hid dim]
		#hidden = [n layers, batch size, hid dim]
		#cell = [n layers, batch size, hid dim]
		
		prediction = self.fc_out(output.squeeze(0))
		
		#prediction = [batch size, output dim]
		
		return prediction, hidden, cell

class Seq2Seq(nn.Module):
	def __init__(self, encoder, decoder, device):
		super().__init__()
		
		self.encoder = encoder
		self.decoder = decoder
		self.device = device
		
		assert encoder.hid_dim == decoder.hid_dim, \
			"Hidden dimensions of encoder and decoder must be equal!"
		assert encoder.n_layers == decoder.n_layers, \
			"Encoder and decoder must have equal number of layers!"
		
	def forward(self, src, trg, teacher_forcing_ratio = 0.5):
		
		#src = [src len, batch size]
		#trg = [trg len, batch size]
		#teacher_forcing_ratio is probability to use teacher forcing
		#e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
		
		batch_size = trg.shape[1]
		trg_len = trg.shape[0]
		trg_vocab_size = self.decoder.output_dim
		
		#tensor to store decoder outputs
		outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
		
		#last hidden state of the encoder is used as the initial hidden state of the decoder
		hidden, cell = self.encoder(src)
		
		#first input to the decoder is the <sos> tokens
		input = trg[0,:]
		
		for t in range(1, trg_len):
			
			#insert input token embedding, previous hidden and previous cell states
			#receive output tensor (predictions) and new hidden and cell states
			output, hidden, cell = self.decoder(input, hidden, cell)
			
			#place predictions in a tensor holding predictions for each token
			outputs[t] = output
			
			#decide if we are going to use teacher forcing or not
			teacher_force = random.random() < teacher_forcing_ratio
			
			#get the highest predicted token from our predictions
			top1 = output.argmax(1) 
			
			#if teacher forcing, use actual next token as next input
			#if not, use predicted token
			input = trg[t] if teacher_force else top1
		
		return outputs

def loadData():
	# endict = open('./E_V/dict.en-vi')
	# trainViet = open('./E_V/train.vi')
	# trainEN = open('./E_V/train.vi')
	# print(endict.readline())
	# return endict

	myData = TranslationDataset('./E_V/train', 
							('.en','.vi'), (SRC,TRG))

	train_data, test_data = myData.splits(exts = ('.en', '.vi'),
							fields = (SRC,TRG), 
							path="./E_V/", 
							train='train',
							validation=None,
							test='tst2012')
	vocabData = TranslationDataset('./E_V/vocab',('.en','.vi'), (SRC,TRG))


	# fileen = open('./E_V/vocab.en','r')
	# filevi = open('./E_V/vocab.vi','r')
	enVec = 0 #Vectors('./E_V/vocab.en')
	viVec = 0 #Vectors('./E_V/vocab.vi')
	# fileen.close()
	# filevi.close()
	# en_vocab, vi_vocab = myData.splits(exts = ('.en','.vi'),
	# 						fields = (SRC,TRG),
	# 						path = "./E_V/",
	# 						train = 'vocab',
	# 						validation = None,
	# 						test = None)
	return train_data, test_data, vocabData

def tokenize_en(text):
	print("TODO")
	return []

def tokenize_vi(text):
	print("TODO")
	return []

def init_weights(m):
	for name, param in m.named_parameters():
		nn.init.uniform_(param.data, -0.08, 0.08)


def trainStep(model, iterator, optimizer, criterion, clip):
	
	epoch_loss = 0
	
	for i, batch in enumerate(iterator):
		
		src = batch.src
		trg = batch.trg
		
		optimizer.zero_grad()
		
		output = model(src, trg)
		
		#trg = [trg len, batch size]
		#output = [trg len, batch size, output dim]
		
		output_dim = output.shape[-1]
		
		output = output[1:].view(-1, output_dim)
		trg = trg[1:].view(-1)
		
		#trg = [(trg len - 1) * batch size]
		#output = [(trg len - 1) * batch size, output dim]
		
		loss = criterion(output, trg)
		
		loss.backward()
		
		torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
		
		optimizer.step()
		
		epoch_loss += loss.item()
		
	return epoch_loss / len(iterator)


def train(model, train_iterator):
	model.apply(init_weights)
	model.train()
	optimizer = torch.optim.Adam(model.parameters())
	TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

	criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)

	N_EPOCHS = 10
	CLIP = 1

	best_valid_loss = float('inf')

	for epoch in range(N_EPOCHS):
		
		start_time = time.time()
		
		train_loss = trainStep(model, train_iterator, optimizer, criterion, CLIP)
		
		end_time = time.time()
		
		epoch_mins, epoch_secs = epoch_time(start_time, end_time)
		
		if valid_loss < best_valid_loss:
			best_valid_loss = valid_loss
			print("Training terminated. Saving model...")
			torch.save(model.state_dict(), './model/tut1-model.pt')
		
		print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
		print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
		print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

def evaluate(model, iterator, criterion):
	
	model.eval()
	
	epoch_loss = 0
	
	with torch.no_grad():
	
		for i, batch in enumerate(iterator):

			src = batch.src
			trg = batch.trg

			output = model(src, trg, 0) #turn off teacher forcing

			#trg = [trg len, batch size]
			#output = [trg len, batch size, output dim]

			output_dim = output.shape[-1]
			
			output = output[1:].view(-1, output_dim)
			trg = trg[1:].view(-1)

			#trg = [(trg len - 1) * batch size]
			#output = [(trg len - 1) * batch size, output dim]

			loss = criterion(output, trg)
			
			epoch_loss += loss.item()
		
	return epoch_loss / len(iterator)

# def test():
# 	#TODO
# 	# load model
# 	print("Loading Model...")
# 	# net = Net()
# 	# net.load_state_dict(torch.load("./model/lstm.pt"))
# 	# net.eval()
# 	# perform translation
# 	# Display average BLEU score with smoothing method 1. No less than 0.07 (7%)
# 	print("test")
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def translate():
	#TODO
	print("Loading Model...")
	net = Net()
	net.load_state_dict(torch.load("./model/lstm.pt"))
	net.eval()
	print("In Translate Mode, Use CTL + C to exit")
	while(1):
		inString = input("> ")
		print(inString)
	# load model
	# accept user input
	# print output
	print("translate")



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('input')
	args = parser.parse_args()
	
	# print(viet_train.read())
	
	train_data, test_data, vocabData = loadData()
	print(vocabData)

	print(f"Number of training examples: {len(train_data)}")
	print(f"Number of testing examples: {len(test_data)}")
	print(vars(train_data.examples[0]))
	SRC.build_vocab(vocabData, min_freq = 1)
	TRG.build_vocab(vocabData, min_freq = 1)
	print(f"ENC_VOCAB: {len(SRC.vocab)}")
	print(f"DEC_VOCAB: {len(TRG.vocab)}")
	print(SRC.vocab)
	INPUT_DIM = len(SRC.vocab)
	OUTPUT_DIM = len(TRG.vocab)
	ENC_EMB_DIM = 256
	DEC_EMB_DIM = 256
	HID_DIM = 512
	N_LAYERS = 1
	ENC_DROPOUT = 0.5
	DEC_DROPOUT = 0.5
	BATCH_SIZE = 32

	train_iterator, test_iterator = BucketIterator.splits(
		(train_data, test_data), 
		batch_size = BATCH_SIZE, 
		device = device)

	# print(train_iterator, test_iterator)
	# print("Bucket: ")
	# print("Number of samples in each bucket: ")
	# print("Bucket scale: ")


	print("Loading Model...")


	enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
	dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
	model = Seq2Seq(enc, dec, device).to(device)	



	if args.input == "train":
		print("train")
		train(model, train_iterator)

	if args.input == "test":
		# test()
		print("test")
	if args.input == "translate":
		translate()


















