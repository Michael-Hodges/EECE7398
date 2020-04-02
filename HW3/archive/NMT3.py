import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
import os
import numpy as np
# from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import random
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import time
import math
import matplotlib.ticker as ticker
from nltk.translate.bleu_score import corpus_bleu

def create_vocabs(s):
  word_idx = {'<PAD>':0} # english vocabulary dictionary provides the index for the given word
  idx_word = {0:'<PAD>'} # given the index provides the word
  c=1
  with open(s) as f:
      for line in f:
        idx_word[c] = line[:-1]
        word_idx[line[:-1]]=c
        c+=1
  return(word_idx,idx_word)

eng_idx,idx_eng = create_vocabs("./E_V/vocab.en")
viet_idx,idx_viet = create_vocabs("./E_V/vocab.vi")

print("English vocabulary\n")
for i in range(15):
  print("{}:{}".format(i,idx_eng[i]))

print("\n\nVietnamese vocabulary\n")
for i in range(15):
  print("{}:{}".format(i,idx_viet[i]))

def create_training(s):
  train=[]
  train_len=[]
  with open(s) as f:
    for line in f:
      train.append(idx_eng[2] +" "+ line[:-1] +" "+ idx_eng[3])
      train_len.append(len((idx_eng[2] +" "+ line[:-1] +" "+ idx_eng[3]).split()))
  return train,np.array(train_len)


en,len_en = create_training("./E_V/train.en")
vi,len_vi = create_training('./E_V/train.vi')

max_en=70
max_viet=90 

def sent_to_index(train,max_len,word_idx):
  x_train=[]
  x_train_len=[]
  if isinstance(train,list): # there are many sentences that have the start and stop symbol
    for sent in train: # sent represents a sentence in the list
      trial=[word_idx[word] if word in set(word_idx) else word_idx["<unk>"] for word in sent.split()]
      if len(trial) < max_len: # if the length is less than max fixed length then we pad and store the original length for packing
        x_train_len.append(len(trial))
        trial.extend(list(np.zeros(max_len-len(trial)).astype(int)))
        x_train.append(trial)
      else:
        x_train_len.append(max_len)
        x_train.append(trial[:max_len]) # we will only consider words upto the position of max_len
    
  if isinstance(train,str):
    trial=[word_idx[word] if word in set(word_idx) else word_idx["<unk>"] for word in train.split()]
    if len(trial) < max_len: # if the length is less than max fixed length then we pad and store the original length for packing
        x_train_len.append(len(trial))
        trial.extend(list(np.zeros(max_len-len(trial)).astype(int)))
        x_train.append(trial)
    else:
        x_train_len.append(max_len)
        x_train.append(trial[:max_len]) # we will only consider words upto the position 
  return np.array(x_train),np.array(x_train_len)


x_train,x_train_len=sent_to_index(en,max_en,eng_idx)
y_train,y_train_len=sent_to_index(vi,max_viet,viet_idx)

np.save('./save/x_train.npy',x_train)
np.save('./save/x_train_len.npy',x_train_len)
np.save('./save/y_train.npy',y_train)
np.save('./save/y_train_len.npy',y_train_len)


X_train=np.load('./save/x_train.npy')
X_train_len=np.load('./save/x_train_len.npy')
Y_train=np.load('./save/y_train.npy')
Y_train_len=np.load('./save/y_train_len.npy')
x_train,x_test,y_train,y_test,x_train_len,x_test_len,y_train_len,y_test_len=train_test_split(X_train,Y_train,X_train_len,Y_train_len,test_size=0.2,random_state=101)


class MyData(Dataset):
    def __init__(self, X,X_len ,Y,Y_len):
        self.data = X
        self.target = Y
        
        self.length1 =X_len
        self.length2 =Y_len

        
        x = self.data[index]
        y = self.target[index]
        x_len = self.length1[index]
        y_len = self.length2[index]
        return x,y,x_len,y_len
    
    def __len__(self):
        return len(self.data)

train = MyData(x_train,x_train_len,y_train,y_train_len)
test = MyData(x_test,x_test_len,y_test,y_test_len)
train_loader=torch.utils.data.DataLoader(train,batch_size=64,shuffle=False,drop_last=True)
test_loader=torch.utils.data.DataLoader(test,batch_size=64,shuffle=False,drop_last=True)


vocab_en=len(eng_idx)
vocab_vi=len(viet_idx)


def sort_batch(X, y, lengths):
    lengths, indx = lengths.sort(dim=0, descending=True)
    X = X[indx]
    y = y[indx]
    return X, y, lengths 

def sort_batch_new(X, lengths):
    lengths, indx = lengths.sort(dim=0, descending=True)
    X = X[indx]
    return X, lengths 


len(train_loader),len(test_loader)

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
    
    packed = torch.nn.utils.rnn.pack_padded_sequence(x, len_x, batch_first=True)
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



class Model(nn.Module):
    def __init__(self,vocab_en,vocab_vi,emb_dim,hid_dim):
        super(Model, self).__init__()
        self.encoder = Encoder(vocab_en,emb_dim,hid_dim)
        self.decoder = Decoder(vocab_vi,emb_dim,hid_dim)
        self.dec_vocab_len = vocab_vi

        self.encoder = self.encoder.to(device)
        self.decoder = self.decoder.to(device)
    
    def forward(self,x_train,x_len,y_train,mask,teacher_forcing_ratio = 0.5):

        enc_op,enc_hid=model.encoder(x_train,x_len)
        # enc_hid.shape : (batch_size,hidden_dim)
        # enc_out.shape : (batch_size,long_seq_length,2*hidden_dim)
        
        #To store the outputs of the decoder
        outputs = torch.zeros(y_train.shape[0],y_train.shape[1],self.dec_vocab_len).to(device)
        # outputs shape : (bz,sq,vocab_viet)

        dec_in = y_train[:,0].unsqueeze(1)
        # dec_in: (bz,1)
        dec_hid = enc_hid
        for t in range(1,y_train.shape[1]):
            if len(dec_hid.shape)==3:
              dec_hid = dec_hid.squeeze(0)
            #print("Dec_in:{}".format(dec_in.shape))
            dec_out,dec_hid,_ = model.decoder(enc_op,dec_hid,dec_in,mask)
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

model=Model(vocab_en,vocab_vi,256,512)

optimizer = optim.Adam(model.parameters(), 
                       lr=0.0005)
criterion = nn.CrossEntropyLoss(ignore_index = 0)

def train(model, train_loader, optimizer, criterion, clip):
    
    model.train()
    
    epoch_loss = 0
    
    for batch,(x_train,y_train,x_len,_) in enumerate(train_loader):

        x_train,y_train,x_len=sort_batch(x_train,y_train,x_len)
        mask=torch.IntTensor((x_train!=0).numpy().astype(int)).to(device)
        x_train,x_len,y_train=x_train.to(device),x_len.to(device),y_train.to(device)
        
        
        optimizer.zero_grad()
        
        output = model(x_train,x_len,y_train,mask,0.5)
        
        
        
        output_dim = output.shape[-1]
        
        output = output[:,1:].reshape(-1, output_dim)
        y_train = y_train[:,1:].reshape(-1)
        
        #trg = [(trg len - 1) * batch size]
        #output = [(trg len - 1) * batch size, output dim]
        
        loss = criterion(output, y_train)
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(train_loader)

def evaluate(model, test_loader, criterion):
    
    model.eval()
    
    epoch_loss = 0
    
    with torch.no_grad():
    
        for batch,(x_train,y_train,x_len,_) in enumerate(test_loader):

            x_train,y_train,x_len=sort_batch(x_train,y_train,x_len)
            mask=torch.IntTensor((x_train!=0).numpy().astype(int)).to(device)
            x_train,x_len,y_train=x_train.to(device),x_len.to(device),y_train.to(device)

            
            optimizer.zero_grad()
        
            output = model(x_train,x_len,y_train,mask,0)
        
            
            output_dim = output.shape[-1]
        
            output = output[:,1:].reshape(-1, output_dim)
            y_train = y_train[:,1:].reshape(-1)

  

            loss = criterion(output, y_train)

            epoch_loss += loss.item()
        
    return epoch_loss / len(test_loader)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


_EPOCHS = 10

best_valid_loss = test_loss

for epoch in range(6,6+N_EPOCHS):
    
    start_time = time.time()
    

    train_loss = train(model, train_loader, optimizer, criterion, 1)
    valid_loss = evaluate(model, test_loader, criterion)
    
    
    end_time = time.time()
    
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), './model/model.pt')
    
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Perplexity: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Perplexity: {math.exp(valid_loss):7.3f}')




















