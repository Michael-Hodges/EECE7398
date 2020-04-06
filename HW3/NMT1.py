import argparse
import os
import random
import sys
import time
from collections import deque

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

import torchtext
from torchtext.data import BucketIterator, Field

import random

import pdb


EN_VOCAB_PATH = './E_V/vocab.en'
VI_VOCAB_PATH = './E_V/vocab.vi'
EN_TRAIN_PATH = './E_V/train.en'
VI_TRAIN_PATH = './E_V/train.vi'
EN_TEST_PATH = './E_V/tst2012.en' #'./E_V/tst2013.en'
VI_TEST_PATH  = './E_V/tst2012.vi' #'./E_V/tst2013.vi'
EN_VALID_PATH = './E_V/tst2013.en'
VI_VALID_PATH = './E_V/tst2013.vi'

BATCH_SIZE = 128

BUCKETS = [(10,10), (14,14), (18,18), (24,24), (33,33), (70,90)]

ENG_MAX_LEN = 70
VI_MAX_LEN = 90

ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5
CLIP = 1
N_EPOCHS = 50

DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# SRC = Field(
# 			sequential=False, use_vocab=False, init_token=None, eos_token=None,
# 			fix_length=None, dtype=torch.int64, preprocessing=None, postprocessing=None,
# 			lower=False, tokenize=None, tokenizer_language=None, include_lengths=False,
# 			batch_first=False, pad_token='<pad>', unk_token=None, pad_first=False,
# 			truncate_first=False, stop_words=None, is_target=False
# 			)
# TRG = Field(
# 			sequential=False, use_vocab=False, init_token=None, eos_token=None,
# 			fix_length=None, dtype=torch.int64, preprocessing=None, postprocessing=None,
# 			lower=False, tokenize=None, tokenizer_language=None, include_lengths=False,
# 			batch_first=False, pad_token='<pad>', unk_token=None, pad_first=False,
# 			truncate_first=False, stop_words=None, is_target=False
# 	)


class Encoder(nn.Module):
	def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
		super().__init__()

		self.hid_dim = hid_dim
		self.n_layers = n_layers

		self.embedding = nn.Embedding(input_dim, emb_dim)

		self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, batch_first=True)

		self.dropout = nn.Dropout(dropout)


	def forward(self, src, src_len):

		#src = [batch size, src len]
		# print("source shape: {}".format(src.shape))
		# print("source len shape: {}".format(src_len.shape))
		embedded = self.dropout(self.embedding(src))
		# print(embedded.shape)
		#embedded = [batch size, src len, emb dim]

		embedded_packed = nn.utils.rnn.pack_padded_sequence(embedded, src_len, batch_first=True, enforce_sorted=False)

		outputs, (hidden, cell) = self.rnn(embedded_packed)

		outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True, padding_value=0, total_length=70)
		# print(outputs.shape)
		# print(hidden.shape)
		# print(cell.shape)
		#outputs = [batch size, src len, hid dim * n directions]
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

		self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, batch_first=False)

		self.fc_out = nn.Linear(hid_dim, output_dim)

		# self.fc_out2 = nn.Linear(1024, output_dim)

		self.dropout = nn.Dropout(dropout)


	def forward(self, input, hidden, cell):

		#input = [batch size]
		#hidden = [n layers * n directions, batch size, hid dim]
		#cell = [n layers * n directions, batch size, hid dim]

		#n directions in the decoder will both always be 1, therefore:
		#hidden = [n layers, batch size, hid dim]
		#context = [n layers, batch size, hid dim]
		input = input.unsqueeze(0)
		embedded = self.dropout(self.embedding(input))
		#embedded = [1, batch size, emb dim]
		output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
		#output = [batch size, seq len, hid dim * n directions]
		#hidden = [n layers * n directions, batch size, hid dim]
		#cell = [n layers * n directions, batch size, hid dim]

		#seq len and n directions will always be 1 in the decoder, therefore:
		#output = [1, batch size, hid dim]
		#hidden = [n layers, batch size, hid dim]
		#cell = [n layers, batch size, hid dim]
		# prediction = F.relu(self.)
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

	def forward(self, src, trg, src_len, teacher_forcing_ratio = 0.5):

		#src = [batch size, src len]
		#trg = [batch size, trg len]
		#teacher_forcing_ratio is probability to use teacher forcing
		#e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
		# print(trg.shape)
		batch_size = trg.shape[0]
		trg_len = trg.shape[1]
		trg_vocab_size = self.decoder.output_dim

		#tensor to store decoder outputs
		outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
		# print(outputs.shape)
		#last hidden state of the encoder is used as the initial hidden state of the decoder
		hidden, cell = self.encoder(src, src_len)

		#first input to the decoder is the <sos> tokens
		input = trg[:,0]
		# print(input.shape)

		for t in range(1, trg_len):

			#insert input token embedding, previous hidden and previous cell states
			#receive output tensor (predictions) and new hidden and cell states

			output, hidden, cell = self.decoder(input, hidden, cell)

			#place predictions in a tensor holding predictions for each token
			outputs[:,t] = output

			#decide if we are going to use teacher forcing or not
			teacher_force = random.random() < teacher_forcing_ratio

			#get the highest predicted token from our predictions
			top1 = output.argmax(1)

			#if teacher forcing, use actual next token as next input
			#if not, use predicted token
			input = trg[:,t] if teacher_force else top1

		return outputs

def init_weights(m):
	for name, param in m.named_parameters():
		nn.init.uniform_(param.data, -0.08, 0.08)

def count_parameters(model):
	return sum(p.numel() for p in model.parameters() if p.requires_grad)

############### DATA Functions ###################
def get_lines():
	# Create dicts used when calling token2id
	id2line_en = {}
	id2line_vi = {}
	id2line_test_en = {}
	id2line_test_vi = {}
	id2line_valid_en = {}
	id2line_valid_vi = {}
	dicts = [id2line_en, id2line_vi, id2line_test_en, id2line_test_vi, id2line_valid_en, id2line_valid_vi]
	paths = [EN_TRAIN_PATH, VI_TRAIN_PATH, EN_TEST_PATH, VI_TEST_PATH, EN_VALID_PATH, VI_VALID_PATH]

	for ii, path in enumerate(paths): # go through all paths in this case english train and viet train
		with open(path, 'r') as f:
			for jj, line in enumerate(f):
				parts = ['<s>']
				line.split()
				parts.extend(line.split() + ['</s>'])
				dicts[ii][jj] = parts # store each sentence in the appropriate dictionary dicts[dict choice][line number]
	return id2line_en, id2line_vi, id2line_test_en, id2line_test_vi, id2line_valid_en, id2line_valid_vi


def line_token2id(line,vocab):
	# takes a list of tokens and converts it to a list of ids using vocab
	id_line = []
	for i in line:
		# print(i)
		tmp = vocab.get(i)
		if tmp == None:
			id_line.append(vocab.get('<unk>'))
		else:
			id_line.append(tmp)
	return id_line, len(id_line)

def line_id2token(line,vocab):
	# takes a list of ids and convert it to token form
	tmp = []
	for i in line:
		tmp.append(vocab[i])
	return tmp

def token2id(batch, vocab):
	# print(batch)
	id_list = [[] for i in range(len(batch))]
	len_list = [0 for i in range(len(batch))]
	max_len = 0
	for ii in batch:
		# id_list[ii] = []
		for jj in batch[ii]:
			tmp = vocab.get(jj)
			if tmp == None:
				id_list[ii].append(vocab.get('<unk>'))
			else:
				id_list[ii].append(tmp)

		len_list[ii] = len(id_list[ii])
		if len_list[ii]>max_len:
			max_len = len_list[ii]
			# id_list[ii].append(jj)'
		# tmp_len = len(batch[ii])
		# if max_len<tmp_len:
		# 	max_len = tmp_len
		# print(len_list[ii])
	return id_list, max_len, len_list

def id2token(batch, vocab):
	token_list = [[] for i in range(len(batch))]
	for ii, id in enumerate(batch):
		for jj in batch[ii]:
			token_list[ii].append(vocab[jj])
	return token_list

def load_vocab(vocab_path):
	# returns list of words as well as dictionary for converting words into numbers
	words = ['<pad>']
	# print(words)
	with open(vocab_path, 'r') as f:
		words1 = f.read().splitlines()
		# words = words.append(f.read().splitlines())
	words.extend(words1)
	return words, {words[i]: i for i in range(len(words))}

def load_data(enc_list, dec_list, max_training_size=None):
	data_buckets = [[] for _ in BUCKETS]
	ii = 0
	for enc, dec in zip(enc_list, dec_list):
		# print("ENC: {}".format(enc))
		# print("DEC: {}".format(dec))

		for bucket_id, (encode_max_size, decode_max_size) in enumerate(BUCKETS):
			# print(bucket_id, encode_max_size, decode_max_size)
			if len(enc)<=encode_max_size:
				data_buckets[bucket_id].append([enc, dec])
				break
	return data_buckets

def load_data_nb(enc_list, dec_list, enc_len, dec_len):
	enc_pad = []
	dec_pad = []
	enc_len_ret = []
	dec_len_ret = []
	ii = 0
	for enc, dec, enc_lens, dec_lens in zip(enc_list, dec_list, enc_len, dec_len):
		if len(enc)<=ENG_MAX_LEN and len(dec)<=VI_MAX_LEN:
			enc.extend(np.zeros(ENG_MAX_LEN-len(enc),dtype=int))
			dec.extend(np.zeros(VI_MAX_LEN-len(dec),dtype=int))
			enc_pad.append(enc)
			dec_pad.append(dec)
			enc_len_ret.append(enc_lens)
			dec_len_ret.append(dec_lens)

	return enc_pad, dec_pad, enc_len_ret, dec_len_ret

def _get_buckets(bucket):
	bucket_sizes = [len(bucket[b]) for b in range(len(BUCKETS))]
	total_samples = sum(bucket_sizes)
	# print(total_samples)
	print("Number of samples in each bucket: {}".format(bucket_sizes))
	# print("Bucket scale: {}".format(np.cumsum(bucket_sizes/total_samples)))
	print(np.cumsum(bucket_sizes))
	buckets_scale = [sum(bucket_sizes[:i+1]) / total_samples for i in range(len(bucket_sizes))]
	print("Bucket scale: {}".format(buckets_scale))
	return buckets_scale

def _get_random_bucket(bucket_scale):
	rand = random.random()
	return min([i for i in range(len(bucket_scale))
		if bucket_scale[i] > rand])


def get_batch(data_bucket, bucket_id, batch_size=1):
	print()

class MyData(torch.utils.data.Dataset):
	def __init__(self, x, y, x_len, y_len):
		self.source = x
		self.target = y
		self.scr_len = x_len
		self.trg_len = y_len

	def __getitem__(self, index):
		# x = torch.stack(self.source[index])
		# y = torch.stack(self.target[index])
		# x = torch.stack(self.source[index])
		x = self.source[index]
		y = self.target[index]
		x_len = self.scr_len[index]
		y_len = self.trg_len[index]

		return x, y, x_len, y_len
	def __len__(self):
		return len(self.source)

def dataLoader(trn_tst):
	print("Processing Data")
	en_words, en_indx = load_vocab('./E_V/vocab.en')
	vi_words, vi_indx = load_vocab('./E_V/vocab.vi')
	# print(en_words[0:20])
	# print(en_indx['<unk>'])
	# print("ENC_VOCAB: {}".format(len(en_words)))
	# print("DEC_VOCAB: {}".format(len(vi_words)))
	# print("Bucket: {}".format(BUCKETS))

	en_train, vi_train, en_test, vi_test, en_valid, vi_valid = get_lines()
	if trn_tst == "train":
		en_identified, max_en, en_len = token2id(en_train, en_indx)
		vi_identified, max_vi, vi_len = token2id(vi_train, vi_indx)
	if trn_tst == "test":
		en_identified, max_en, en_len = token2id(en_test, en_indx)
		vi_identified, max_vi, vi_len = token2id(vi_test, vi_indx)
	if trn_tst == "train_eval":
		en_identified, max_en, en_len = token2id(en_test, en_indx)
		vi_identified, max_vi, vi_len = token2id(vi_test, vi_indx)

	# tokenized = id2token(identified, words)
	assert len(en_identified) == len(vi_identified), "Translation data not the same length"
	# data_buckets = load_data(en_identified, vi_identified)
	enc_data, dec_data, en_len, vi_len = load_data_nb(en_identified, vi_identified, en_len, vi_len)
	# bucket_scale = _get_buckets(data_buckets)
	print("Loading Model...")
	return enc_data, dec_data, len(en_words), len(vi_words), en_len, vi_len, en_words, en_indx, vi_words, vi_indx
	# torchtext.data.Dataset(en_identified, vi_identified, )
	# torchtext.data.Example.fromlist((en_identified[0], vi_identified[0]), (SRC, TRG))
	# for ii, sample in enumerate(train_iterator):
		# print(ii + ":" + sample)

def train_step(model, data_iter, optimizer, criterion, clip):
	model.train()
	epoch_loss = 0
	for i, (enc,dec, src_len, trg_len) in enumerate(data_iter):
		src = enc
		trg = dec

		optimizer.zero_grad()

		output = model(src,trg, src_len)
		# print("shape from model: {}".format(output.shape))
		# trg = [batch size, trg len]
		# output = [batch, trg len, output dim]

		output_dim  = output.shape[-1]
		# print("output dimension: {}".format(output_dim))
		output = output[1:].view(-1, output_dim)
		trg = trg[1:].view(-1)
		# print("output shape: {}".format(output.shape)) #[5670, 7710]
		# print("target shape:{}".format(trg.shape))	   # [5670]

		loss = criterion(output,trg)

		loss.backward()

		torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

		optimizer.step()

		epoch_loss += loss.item()

		# print("Batch: {}/{}, Loss:{}".format(i,len(data_iter), loss.item()))

	return epoch_loss/len(data_iter)


def eval_train(model, data_iter, criterion):
	model.eval()
	epoch_loss = 0
	with torch.no_grad():
		for i, (enc,dec, src_len, trg_len) in enumerate(data_iter):
			src = enc
			trg = dec

			output = model(src,trg, src_len, 0)
			# print("shape from model: {}".format(output.shape))
			# trg = [batch size, trg len]
			# output = [batch, trg len, output dim]

			output_dim  = output.shape[-1]
			# print("output dimension: {}".format(output_dim))
			output = output[1:].view(-1, output_dim)
			trg = trg[1:].view(-1)
			# print("output shape: {}".format(output.shape)) #[5670, 7710]
			# print("target shape:{}".format(trg.shape))	   # [5670]

			loss = criterion(output,trg)
			epoch_loss += loss.item()
	return epoch_loss/len(data_iter)

def epoch_time(start_time, end_time):
	elapsed_time = end_time - start_time
	elapsed_mins = int(elapsed_time / 60)
	elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
	return elapsed_mins, elapsed_secs

def train():
	# data_bucket, bucket_scale, len_enc_voc, len_dec_voc = dataLoader()
	enc_data, dec_data, len_enc_voc, len_dec_voc, enc_sent_len, dec_sent_len, _, _, vi_words, _ = dataLoader("train")
	enc_val_data, dec_val_data, _, _, enc_val_sent_len, dec_val_sent_len, _, _, _, _ = dataLoader("train_eval")
	INPUT_DIM = len_enc_voc
	OUTPUT_DIM = len_dec_voc


	enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
	dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
	model = Seq2Seq(enc, dec, DEVICE).to(DEVICE)
	# print(f'The model has {count_parameters(model):,} trainable parameters')

	enc_tens = torch.tensor(enc_data, dtype = torch.int64).to(DEVICE)
	dec_tens = torch.tensor(dec_data, dtype = torch.int64).to(DEVICE)
	train_dataset = MyData(enc_tens, dec_tens, enc_sent_len, dec_sent_len)
	# print(max(enc_sent_len))
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle=False, drop_last = True)

	enc_val_tens = torch.tensor(enc_val_data, dtype = torch.int64).to(DEVICE)
	dec_val_tens = torch.tensor(dec_val_data, dtype = torch.int64).to(DEVICE)
	valid_dataset = MyData(enc_val_tens, dec_val_tens, enc_val_sent_len, dec_val_sent_len)
	valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size = BATCH_SIZE, shuffle=False, drop_last = True)
	# out = model(enc_sample.to(DEVICE), dec_sample.to(DEVICE))

	model.apply(init_weights)
	optimizer = optim.Adam(model.parameters(), 0.001)
	criterion = nn.CrossEntropyLoss(ignore_index=0)

	best_valid_loss = float('inf')
	best_valid_BLEU = float(0)
	for epoch in range(N_EPOCHS):
		data_iter = iter(train_loader)
		valid_iter = iter(valid_loader)
		bleu_iter = iter(valid_loader)
		start_time = time.time()
		train_loss = train_step(model, data_iter, optimizer, criterion, CLIP)
		valid_loss = eval_train(model, valid_iter, criterion)
		bleu_score = evaluate_step(model, bleu_iter, criterion, vi_words)
		# for i, (enc,dec, src_len, trg_len) in enumerate(data_iter):
		if bleu_score > best_valid_BLEU:
			best_valid_BLEU = bleu_score
			torch.save(model.state_dict(), './model/nmt_5.pt')
		end_time = time.time()
		epoch_min, epoch_seconds = epoch_time(start_time, end_time)
		print("Epoch: {} | Train Loss: {}| Valid Loss: {} | BLEU: {}".format(epoch, train_loss, valid_loss, bleu_score))

	print("Training Terminated. Saving model...")
	# torch.save(model.state_dict(), './model/nmt_5.pt')


def evaluate_step(model,iterator,criterion, viet_words):
	model.eval()
	average_bleu = 0
	smoothing = SmoothingFunction(epsilon = 0.1, alpha=5, k=5)

	with torch.no_grad():
		for i, (enc,dec, src_len, trg_len) in enumerate(iterator):
			src = enc
			trg = dec

			output = model(src,trg, src_len, 0)
			output = output.argmax(dim=2)

			# print(output[0])
			# print("model output shape: {}".format(output.shape))
			# print("target shape: {}".format(trg.shape))

			word_output = [[] for k in src_len]
			batch_output = output.tolist()
			correct_output = trg.tolist()
			print(batch_output[0:5])
			for ii, (j, jj) in enumerate(zip(batch_output, correct_output)):
				word_output[ii] = depad(line_id2token(j, viet_words))
				correct_output[ii] = depad(line_id2token(jj, viet_words))
			# print(correct_output)
			# print(word_output)
			# loss = criterion(output,trg)
			# epoch_loss += loss.item()
			average_bleu += corpus_bleu(correct_output, word_output, weights = (1.0, 0.0, 0.0, 0.0), smoothing_function=smoothing.method1, auto_reweigh=False)

			# print("Batch: {}/{}, Loss:{}".format(i,len(data_iter), loss.item()))
	return average_bleu/len(iterator)*100

def test():
	enc_data, dec_data, len_enc_voc, len_dec_voc, enc_sent_len, dec_sent_len, en_words, en_indx, vi_words, vi_indx  = dataLoader("test")
	INPUT_DIM = len_enc_voc
	OUTPUT_DIM = len_dec_voc


	enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
	dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
	model = Seq2Seq(enc, dec, DEVICE).to(DEVICE)

	model.load_state_dict(torch.load('./model/nmt_5.pt'))

	enc_tens = torch.tensor(enc_data, dtype = torch.int64).to(DEVICE)
	dec_tens = torch.tensor(dec_data, dtype = torch.int64).to(DEVICE)
	test_dataset = MyData(enc_tens, dec_tens, enc_sent_len, dec_sent_len)

	test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle=False, drop_last = True)
	data_iter = iter(test_loader)
	criterion = nn.CrossEntropyLoss(ignore_index = 0)

	bleu_score = evaluate_step(model, data_iter, criterion, vi_words)

	print("Average BELU: {}".format(bleu_score))

def translate_one(sentence, s_len, eng_idx, viet_idx, model, max_len = 90):
	model.eval()

	src_tens = torch.LongTensor(sentence).to(DEVICE)
	src_len = torch.LongTensor([s_len]).to(DEVICE)
	src_tens = src_tens.unsqueeze(0)
	with torch.no_grad():
		hidden, cell = model.encoder(src_tens, src_len)
	trg = torch.zeros(src_tens.shape[0], max_len, dtype=torch.long).to(DEVICE)
	trg[:,0] = viet_idx['<s>']
	# print("trg.shape: {}".format(trg.shape))
	# print("trg: {}".format(trg))
	for i in range(max_len):
		with torch.no_grad():
			prediction, hidden, cell = model.decoder(trg[:,i], hidden, cell)
			prediction = prediction.argmax(1).item()
			# print(prediction)
			if prediction == viet_idx['</s>']:
				break
			try:
				trg[:,i+1] = prediction
			except:
				print("Did not reach </s> token before reaching out of index: {} must be < {} \n".format(i+1, max_len))
	# print(trg)
	return trg.squeeze(0)

def pad(sentence, max_len):
	if len(sentence) == max_len:
		return sentence
	num_pads = max_len-len(sentence)
	sentence.extend([0 for i in range(num_pads)])
	return sentence

def depad(sentence):
	stopwords = ['<s>','<pad>','<unk>','</s>']
	return [word for word in sentence if word not in stopwords]


def translate():
	# enc_data, dec_data, len_enc_voc, len_dec_voc, enc_sent_len, dec_sent_len, en_words, en_indx, vi_words, vi_indx = dataLoader("test")
	en_words, en_indx = load_vocab('./E_V/vocab.en')
	vi_words, vi_indx = load_vocab('./E_V/vocab.vi')
	INPUT_DIM = len(en_indx)
	OUTPUT_DIM = len(vi_indx)

	print("Loading Model...")
	enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
	dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
	model = Seq2Seq(enc, dec, DEVICE).to(DEVICE)

	model.load_state_dict(torch.load('./model/nmt_5.pt'))
	print("In Translation Mode. Press ctl+c to exit...")
	while(1):
		to_translate = input(">")
	# to_translate = "And these are the questions that we have to worry about for the next 50 years ."
		to_translate = "<s> "+to_translate+" </s>"
		to_translate = to_translate.split()
		sent_id, sent_len = line_token2id(to_translate,en_indx)
		padded_sent = pad(sent_id, 70)
		translate_indx = translate_one(sent_id,sent_len, en_indx, vi_indx, model, max_len=90)
		translated_tokens = depad(line_id2token(translate_indx, vi_words))
		print(" ".join(translated_tokens))

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('input')
	args = parser.parse_args()

	if args.input == "train":
		train()
	if args.input == "test":
		test()
	if args.input == "translate":
		translate()

if __name__ == '__main__':
	main()
	# dataLoader()
