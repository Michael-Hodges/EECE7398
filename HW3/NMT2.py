import argparse
import os
import random
import sys
import time
from collections import deque

import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim

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

BATCH_SIZE = 64

BUCKETS = [(10,10), (14,14), (18,18), (24,24), (33,33), (70,90)]

ENG_MAX_LEN = 70
VI_MAX_LEN = 90

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
	def __init__(self,vocab_en,emb_dim,hidden_dim):
		super(Encoder,self).__init__()

		self.embeds = nn.Embedding(vocab_en, emb_dim)

		self.GRU = nn.GRU(emb_dim,hidden_dim,num_layers=1,batch_first=True,bidirectional=True)

		self.reduce_hid = nn.Linear(hidden_dim * 2, hidden_dim)

		self.dropout = nn.Dropout(0.5)

	def forward(self,x,len_x):
		# input x has the shape:(batch_size,seq_length)

		x = self.dropout(self.embeds(x)) 
		# after embedding the shape: (batch_size,longest seq_length, emb_dim)

		packed = torch.nn.utils.rnn.pack_padded_sequence(x, len_x, batch_first=True, enforce_sorted=False)
		# packing is done to efficiently run through the different padded instances

		enc_out, enc_hid = self.GRU(packed)
		# enc_hid.shape : (2,batch_size,hidden_dim)Output for the last time step

		enc_out , _ = torch.nn.utils.rnn.pad_packed_sequence(enc_out, batch_first=True,padding_value=0,total_length=70)
		# enc_out.shape : (batch_size,long_seq_length,2*hidden_dim) cause we use a bidirectional GRU
		enc_out = enc_out.contiguous()

		enc_hid = torch.cat(list(enc_hid), dim=1)
		# enc_hid.shape : batch_size,2*hidden_dim
		enc_hid = F.relu(self.reduce_hid(enc_hid))
		# enc_hid.shape : batch_size,hidden_dim


		return enc_out,enc_hid

class Attention(nn.Module):
	def __init__(self,hidden_dim):
		super(Attention,self).__init__()
		# used for attention
		self.W1 = nn.Linear(hidden_dim * 2,hidden_dim )
		self.W2 = nn.Linear(hidden_dim ,hidden_dim )
		self.V = nn.Linear(hidden_dim, 1)

	def forward(self,enc_out,dec_hid,mask):
		#enc_out: encoder output for all the states (bz,seq,2*hid)
		#dec_hid: Hidden state output for the last state (bz,hid) represents the hidden state rep of the decoder

		dec_hid = dec_hid.unsqueeze(1)
		# changes enc_hidden shape to (bz,1,hidden) to match with the enc_out shape
		#print(dec_hid.shape)
		#print(enc_out.shape)

		score = self.V(torch.tanh(self.W1(enc_out) + self.W2(dec_hid))) 
		# Score to check the similarity between the single hidden state component all the hidden states of the encoder output
		# Score should have the shape (bz,sq,hidden)-->(bz,sq,1)
		score = score.squeeze(2)
		# Score should have the shape (bz,sq)

		score = score * mask

		attention_weights = torch.softmax(score, dim=1) 
		# we take the softmax over the first axis which represents the sequence 
		# attention_weights shape (bz,seq) for the single hidden element we know on which input sequence to focus on

		return attention_weights.unsqueeze(2)

class Decoder(nn.Module):
	def __init__(self,vocab_vi,emb_dim,hidden_dim):
		super(Decoder,self).__init__()
		# used for the decoder
		self.x_context = nn.Linear(hidden_dim*2 + emb_dim, emb_dim)

		self.embeds = nn.Embedding(vocab_vi, emb_dim)

		self.GRU = nn.GRU(emb_dim,hidden_dim,num_layers=1,batch_first=True)

		self.out = nn.Linear(hidden_dim , vocab_vi)

		self.attention = Attention(hidden_dim)

		self.dropout = nn.Dropout(0.5)

	def forward(self,enc_out,dec_hid,x,mask):
		#enc_out: encoder output for all the states (bz,seq,2*hid)
		#dec_hid: Hidden state output for the last state (bz,hid) represents the hidden state rep of the decoder
		#x: input to the decoder, mostly the output of the previous time step
		#mask: shape (bz,sq)
		attention_weights = self.attention(enc_out,dec_hid,mask) 
		# attention_weights shape (bz,seq,1)

		context_vector = attention_weights * enc_out
		# contect vector shape : (bz,seq,2*hid)

		context_vector = torch.sum(context_vector, dim=1)
		# contect_vector shape : (bz,2*hid)

		x = self.dropout(self.embeds(x)) 
		# after embedding the shape: (batch_size,1, emb_dim)

		context_vector =  context_vector.unsqueeze(1)
		# context_vector shape:((bz,1,2*hid))

		#print(context_vector.shape,x.shape)
		x = torch.cat((context_vector, x), -1)
		# x shape: (bz,seq,2*hidden+emb)

		x = self.x_context(x)
		# x shape: (bz,seq,emb)

		dec_hid = dec_hid.unsqueeze(0)
		# dec_hid shape: (1,bz,hid)

		dec_out, dec_hid = self.GRU(x,dec_hid)
		# dec_out.shape : batch_size,1,hidden_dim 
		# dec_hid.shape : 1,batch_size,hidden_dim

		dec_out =  dec_out.view(-1, dec_out.size(2))
		# dec_out.shape : batch_size,hidden_dim 


		dec_out = self.out(dec_out)
		# dec_out.shape : batch_size,vocab_dim     

		return dec_out,dec_hid,attention_weights.squeeze(2)

class Seq2Seq(nn.Module):
	def __init__(self,vocab_en,vocab_vi,emb_dim,hid_dim):
		super(Seq2Seq, self).__init__()
		self.encoder = Encoder(vocab_en,emb_dim,hid_dim)
		self.decoder = Decoder(vocab_vi,emb_dim,hid_dim)
		self.dec_vocab_len = vocab_vi

		self.encoder = self.encoder.to(DEVICE)
		self.decoder = self.decoder.to(DEVICE)
	
	def forward(self,x_train,x_len,y_train,mask,teacher_forcing_ratio = 0.5):

		enc_op,enc_hid=self.encoder(x_train,x_len)
		# enc_hid.shape : (batch_size,hidden_dim)
		# enc_out.shape : (batch_size,long_seq_length,2*hidden_dim)
		
		#To store the outputs of the decoder
		outputs = torch.zeros(y_train.shape[0],y_train.shape[1],self.dec_vocab_len).to(DEVICE)
		# outputs shape : (bz,sq,vocab_viet)

		dec_in = y_train[:,0].unsqueeze(1)
		# dec_in: (bz,1)
		dec_hid = enc_hid
		for t in range(1,y_train.shape[1]):
			if len(dec_hid.shape)==3:
			  dec_hid = dec_hid.squeeze(0)
			#print("Dec_in:{}".format(dec_in.shape))
			dec_out,dec_hid,_ = self.decoder(enc_op,dec_hid,dec_in,mask)
			#dec_hid = dec_hid.squeeze(0)
			outputs[:,t,:] = dec_out

			#decide if we are going to use teacher forcing or not
			teacher_force = random.random() < teacher_forcing_ratio
			
			#get the highest predicted token from our predictions
			top1 = dec_out.argmax(1) 
			
			#if teacher forcing, use actual next token as next input
			#if not, use predicted token
			dec_in = y_train[:,t] if teacher_force else top1
			dec_in = dec_in.unsqueeze(1)
			

		
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
	dicts = [id2line_en, id2line_vi, id2line_test_en, id2line_test_vi]	
	paths = [EN_TRAIN_PATH, VI_TRAIN_PATH, EN_TEST_PATH, VI_TEST_PATH]

	for ii, path in enumerate(paths): # go through all paths in this case english train and viet train
		with open(path, 'r') as f:
			for jj, line in enumerate(f):
				parts = ['<s>']
				line.split()
				parts.extend(line.split() + ['</s>'])
				dicts[ii][jj] = parts # store each sentence in the appropriate dictionary dicts[dict choice][line number]
	return id2line_en, id2line_vi, id2line_test_en, id2line_test_vi

def line_token2id(line,vocab):
	# takes a list of tokens and converts it to a list of ids using vocab
	id_line = []
	for i in line:
		tmp = vocab.get(i)
		if tmp == None:
			id_line.append(vocab.get('<unk>'))
		else:
			id_line.append(tmp)
	return id_line

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
	print("ENC_VOCAB: {}".format(len(en_words)))
	print("DEC_VOCAB: {}".format(len(vi_words)))
	# print("Bucket: {}".format(BUCKETS))

	en_train, vi_train, en_test, vi_test = get_lines()
	if trn_tst == "train":
		en_identified, max_en, en_len = token2id(en_train, en_indx)
		vi_identified, max_vi, vi_len = token2id(vi_train, vi_indx)
	if trn_tst == "test":
		en_identified, max_en, en_len = token2id(en_test, en_indx)
		vi_identified, max_vi, vi_len = token2id(vi_test, vi_indx)
	print(len(en_identified))
	# tokenized = id2token(identified, words)
	assert len(en_identified) == len(vi_identified), "Translation data not the same length"
	# data_buckets = load_data(en_identified, vi_identified)
	enc_data, dec_data, en_len, vi_len = load_data_nb(en_identified, vi_identified, en_len, vi_len)
	# bucket_scale = _get_buckets(data_buckets)
	print("Loading Model...")
	return enc_data, dec_data, len(en_words), len(vi_words), en_len, vi_len
	# torchtext.data.Dataset(en_identified, vi_identified, )
	# torchtext.data.Example.fromlist((en_identified[0], vi_identified[0]), (SRC, TRG))
	# for ii, sample in enumerate(train_iterator):
		# print(ii + ":" + sample)

def train_step(data_iter):
	print()

def epoch_time(start_time, end_time):
	elapsed_time = end_time - start_time
	elapsed_mins = int(elapsed_time / 60)
	elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
	return elapsed_mins, elapsed_secs

def train():
	# data_bucket, bucket_scale, len_enc_voc, len_dec_voc = dataLoader()
	enc_data, dec_data, len_enc_voc, len_dec_voc, enc_sent_len, dec_sent_len = dataLoader("train")
	INPUT_DIM = len_enc_voc
	OUTPUT_DIM = len_dec_voc
	ENC_EMB_DIM = 256
	DEC_EMB_DIM = 256
	HID_DIM = 512
	N_LAYERS = 2
	ENC_DROPOUT = 0.5
	DEC_DROPOUT = 0.5
	CLIP = 1
	N_EPOCHS = 10

	# enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
	# dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
	model = Seq2Seq(INPUT_DIM, OUTPUT_DIM, ENC_EMB_DIM, HID_DIM).to(DEVICE)
	print(f'The model has {count_parameters(model):,} trainable parameters')

	enc_tens = torch.tensor(enc_data, dtype = torch.int64).to(DEVICE)
	dec_tens = torch.tensor(dec_data, dtype = torch.int64).to(DEVICE)
	train_dataset = MyData(enc_tens, dec_tens, enc_sent_len, dec_sent_len)
	print(max(enc_sent_len))
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle=False, drop_last = True)

	data_iter = iter(train_loader)
	# enc_sample, dec_sample = data_iter.next()

	# out = model(enc_sample.to(DEVICE), dec_sample.to(DEVICE))


	model.apply(init_weights)
	optimizer = optim.Adam(model.parameters(), lr=0.0005)
	criterion = nn.CrossEntropyLoss(ignore_index = 0)
	
	

	for epoch in range(N_EPOCHS):
		start_time = time.time()
		model.train()
		epoch_loss = 0
		for i, (enc,dec, src_len, trg_len) in enumerate(data_iter):
			src = enc
			trg = dec
			mask=torch.IntTensor((enc!=0).numpy().astype(int)).to(Device)
			optimizer.zero_grad()

			output = model(src,src_len, trg, 0.5)
			# print("shape from model: {}".format(output.shape))
			# trg = [batch size, trg len]
			# output = [batch, trg len, output dim]

			output_dim  = output.shape[-1]
			# print("output dimension: {}".format(output_dim))
			output = output[1:].view(-1, output_dim)
			trg = trg[1:].view(-1)
			# print("output shape: {}".format(output.shape))
			# print("target shape:{}".format(trg.shape))

			loss = criterion(output,trg)

			loss.backward()

			torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)

			optimizer.step()

			epoch_loss += loss.item()

		epoch_loss = epoch_loss/len(data_iter)

		end_time = time.time()
		epoch_min, epoch_seconds = epoch_time(start_time, end_time)
		print("Epoch: {} | Time: {}m {}s".format(epoch, start_time, end_time))
		print("Train Loss: {}".format(epoch_loss))

	print("Training Terminated. Saving model...")
	torch.save(model.state_dict(), './model/nmt.pt')

def test():
	enc_data, dec_data, len_enc_voc, len_dec_voc = dataLoader("test")	
	INPUT_DIM = len_enc_voc
	OUTPUT_DIM = len_dec_voc
	ENC_EMB_DIM = 256
	DEC_EMB_DIM = 256
	HID_DIM = 512
	N_LAYERS = 2
	ENC_DROPOUT = 0.5
	DEC_DROPOUT = 0.5
	CLIP = 1
	N_EPOCHS = 10

	enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
	dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
	model = Seq2Seq(enc, dec, DEVICE).to(DEVICE)

	model.load_state_dict(torch.load('./model/nmt.pt'))

	enc_tens = torch.tensor(enc_data, dtype = torch.int64).to(DEVICE)
	dec_tens = torch.tensor(dec_data, dtype = torch.int64).to(DEVICE)
	test_dataset = MyData(enc_tens, dec_tens)

	test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle=False, drop_last = True)
	data_iter = iter(test_loader)
	criterion = nn.CrossEntropyLoss()
	model.eval()

	epoch_loss = 0

	with torch.no_grad():
		for i, (enc,dec) in enumerate(test_loader):

			src = enc
			trg = dec

			output = model(src, trg, 0)

			output_dim  = output.shape[-1]
			output = output[1:].view(-1, output_dim)
			trg = trg[1:].view(-1)

			loss = criterion(output, trg)

			epoch_loss +=loss.item()
	print("Epoch Loss: {}".format(epoch_loss/len(test_loader)))

def translate():
	print("In Translation Mode. Press ctl+c to exit...")
	print("Loading Model...")
	INPUT_DIM = len_enc_voc
	OUTPUT_DIM = len_dec_voc
	ENC_EMB_DIM = 256
	DEC_EMB_DIM = 256
	HID_DIM = 512
	N_LAYERS = 2
	ENC_DROPOUT = 0.5
	DEC_DROPOUT = 0.5
	CLIP = 1
	N_EPOCHS = 10
	enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
	dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
	model = Seq2Seq(enc, dec, DEVICE).to(DEVICE)

	model.load_state_dict(torch.load('./model/nmt.pt'))
	while(1):
		to_translate = input(">")

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