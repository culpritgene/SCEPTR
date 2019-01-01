#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 19:16:51 2018

@author: lunar
"""


import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F

#######################################################################################################
######################### ALL MODELS USED IN THE ANALYSIS #############################################
##################################___ MODEL 1 ___#######################################################

class LSTM_Kmers(nn.Module):
    """ Bi-LSTM system concatenating embedded kmer representations with LSTM output from direct reader of 
    embedded letters (e.g. amino acids). Concatenation allows for more intricate distance metrics between 
    each individual kmer (we use ither 3mers or 4mers). The model with 4mers learns slowly and requires
    decent GPU. 
    Potential enhancement of results might be obtained with addition of bidirectional layers (although for 
    'word' reader this should be too cumbursome and for letter reader - not very usefull), learning with better
    regularization, especially at the last layer."""
    def __init__(self, embedding_chars, embedding_words, hidden_dim_read, alphabet_size, 
                vocab_size, word_latent_dim, hidden_dim_word, tagset_size, word_to_ix, char_to_ix):
        super( LSTM_Kmers , self).__init__()
        
        self.char_to_ix = char_to_ix
        self.word_to_ix = word_to_ix
        
        self.hidden_dim = {'read': hidden_dim_read, 'word':hidden_dim_word}        
        self.hidden_read = self.init_hidden('read')
        self.hidden_word = self.init_hidden('word')

        self.embed_chars = nn.Embedding(alphabet_size, embedding_chars)
        self.lstm_read = nn.LSTM(embedding_chars, hidden_dim_read, dropout=0.35)
        self.hidden2latent = nn.Linear(hidden_dim_read, word_latent_dim)
        ### here is tanh after linear layes
        ### word part
        self.embed_words = nn.Embedding(vocab_size, embedding_words)
        self.lstm_tag = nn.LSTM(word_latent_dim+embedding_words, hidden_dim_word, dropout=0.35)
        self.hidden2tag = nn.Linear(hidden_dim_word, tagset_size)

        
    def forward(self, sentence):
        embeds_sent = self.embed_words(  prepare_sequence(sentence, self.word_to_ix))
        sentence_outs = []
        for word, embed_word in zip(sentence, embeds_sent):
            embeds_chars = self.embed_chars( prepare_sequence(word, self.char_to_ix))
            
            lstm_char_out, self.hidden_read = self.lstm_read( embeds_chars.view(len(word),1,-1) , self.hidden_read)
            latent_rep = F.tanh(self.hidden2latent( self.hidden_read[0] ))
            lstm_tag_out, self.hidden_word = self.lstm_tag( torch.cat([embed_word.view(1,1,-1), latent_rep], dim=2).view(1,1,-1),
                                                          self.hidden_word)
            sentence_outs.append(lstm_tag_out)
        
        latent_space = torch.cat(sentence_outs).view(len(sentence),-1)
###### this is used to train network on Vdjdb dataset 
#         try:
#             tag_space = self.hidden2tag( torch.cat(sentence_outs).view(len(sentence),-1) )
#         except RuntimeError:
#             return None
#         tag_score = F.log_softmax(tag_space, dim=1)                                   
        return latent_space, sentence_outs
    
    def init_hidden(self, key):
        return (Variable(torch.zeros(1,1, self.hidden_dim[key])), Variable(torch.zeros(1,1, self.hidden_dim[key])))
    
##################################___ MODEL 2 ___#######################################################

class autoencoder_rnn1(nn.Module):
    """ Simplistic lstm reader model which attempts to reconstruct word from the latent representation
     obtained out of terminal hidden layer. Here is no bidirectionality, no dropouts and no other 
     regularization terms. """
    def __init__(self, embedding_dim, hidden_dim, latent_space_dim, embedding_dim_VJ):
        super(autoencoder_rnn1, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.hidden_in = self.init_hidden()
                
        self.embedding_Vg  = nn.Embedding(Vgenes_size, embedding_dim_VJ)
        self.embedding_Jg = nn.Embedding(Jgenes_size, embedding_dim_VJ)
            
        self.embedding  = nn.Embedding(alphabet_size, embedding_dim)
        self.lstm_in = nn.LSTM(embedding_dim, hidden_dim)
        self.latent_space = nn.Linear(hidden_dim, latent_space_dim)
        self.latent_space_data = 0 ### for convenience lets store this in the class instance object
        self.reconstructor = nn.LSTM(embedding_dim, embedding_dim)
        
    
    def encode(self, x):
        embedding = self.embedding(x)
        lstm_out, lstm_in_hidden = self.lstm_in(embedding.view(len(x),1,-1), self.hidden_in)  ### here we translate all sequence at once! 
        latent_space = F.tanh(self.latent_space(lstm_in_hidden[0]))
        return latent_space, embedding
    
    def decode(self, input_length, latent_space): 
        
        previous_letter = torch.LongTensor([all_letters.index('X')]) ### lets try to reversely reconstruct from end of sequence symbol 
        previous_letter = self.embedding(previous_letter).view(1,1,-1)
        
        reconstructed_sequence = [previous_letter]
        lstm_out_hidden = (latent_space.view(1, 1, -1), latent_space.view(1, 1, -1))
       # print( lstm_out_hidden[0].shape )
        for i in range(input_length-2):  ### change this for generator later!
            current_out, lstm_out_hidden = self.reconstructor(previous_letter.view(1, 1, -1), lstm_out_hidden)
            reconstructed_sequence.append(current_out)
            previous_letter = current_out
            
        last_letter = torch.LongTensor([all_letters.index('B')]) ### lets try to reversely reconstruct from end of sequence symbol 
        last_letter = self.embedding(last_letter).view(1,1,-1)
        reconstructed_sequence.append(last_letter)
        reconstructed_sequence = torch.cat(reconstructed_sequence[::-1])
        
        return reconstructed_sequence, lstm_out_hidden
    
    def forward(self, x, Vg, Jg, tick):
    #    if tick > 6:
    #        self.embedding.requires_grad = False
        Vg_emb = self.embedding_Vg(Vg).unsqueeze(0)
        Jg_emb = self.embedding_Jg(Jg).unsqueeze(0)
        latent_space, embedded_input = self.encode(x)
       # print(latent_space.shape, Vg_emb.shape, Jg_emb.shape)
        latent_space = torch.cat( (Vg_emb, latent_space, Jg_emb), dim=2 )  ## new dimention of latent space! 
        reconstructed_sequence, reconstruction_out_hidden = self.decode(len(x), latent_space)
        return latent_space, reconstructed_sequence, embedded_input
        
    def init_hidden(self):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

        
    


##################################___ MODEL 3 ___#######################################################

class VAE(nn.Module):
    """ Variational autoencoder as a first variation for simplistinc lstm-autoencoder v1.
    Unfortunately, I was unable to train it initially. Perhaps I should read more about
    VAEs, come to terms about possible usage of those for such precision-demanding task
    and understand where exactly and based on what distribution should I use reparametrization."""
    
    def __init__(self, embedding_dim, hidden_dim, latent_space_dim,
                 prelatent_dim, postlatent_dim, embedding_dim_VJ):
        super(autoencoder_rnn1, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.hidden_in = self.init_hidden()
                
        self.embedding_Vg  = nn.Embedding(Vgenes_size, embedding_dim_VJ)
        self.embedding_Jg = nn.Embedding(Jgenes_size, embedding_dim_VJ)
            
        self.embedding  = nn.Embedding(alphabet_size, embedding_dim)
        self.lstm_in = nn.LSTM(embedding_dim, hidden_dim)
        
        self.prelatent = nn.Linear(hidden_dim, prelatent_dim)
        self.latent_space_Mu = nn.Linear(hidden_dim, latent_space_dim)
        self.latent_space_Sigma = nn.Linear(hidden_dim, latent_space_dim)

        self.postlatent = nn.Linear(postlatent_dim, embedding_dim)
        self.reconstructor = nn.LSTM(embedding_dim, embedding_dim)
        
    
    def encode(self, x):
        embedding = self.embedding(x)
        lstm_out, lstm_in_hidden = self.lstm_in(embedding.view(len(x),1,-1), self.hidden_in)  ### here we translate all sequence at once! 
       # prelatent_ = F.tanh(self.prelatent(lstm_in_hidden[0])) ### we are using tanh so we should be careful with 
        ### vanishing gradients - why not Relu though?
       # print(prelatent_.shape)
        latent_space_Mu = self.latent_space_Mu(lstm_in_hidden[0])
        latent_space_Sigma = self.latent_space_Sigma(lstm_in_hidden[0])
        return latent_space_Mu, latent_space_Sigma, embedding

    
    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.randn(std.size())**3
        return eps.mul(std).add_(mu)   
    
    
    def decode(self, input_length, latent_space): 
        
        postlatent_ = F.tanh( self.postlatent(latent_space) )
        lstm_out_hidden = (postlatent_.view(1, 1, -1), postlatent_.view(1, 1, -1))
        
        previous_letter = torch.LongTensor([all_letters.index('X')]) ### lets try to reversely reconstruct from end of sequence symbol 
        previous_letter = self.embedding(previous_letter).view(1,1,-1)
        reconstructed_sequence = [previous_letter]
        
       # print( lstm_out_hidden[0].shape )
        for i in range(input_length-2):  ### change this for generator later!
            current_out, lstm_out_hidden = self.reconstructor(previous_letter.view(1, 1, -1), lstm_out_hidden)
            reconstructed_sequence.append(current_out)
            previous_letter = current_out
            
        last_letter = torch.LongTensor([all_letters.index('B')]) ### lets try to reversely reconstruct from end of sequence symbol 
        last_letter = self.embedding(last_letter).view(1,1,-1)
        reconstructed_sequence.append(last_letter)
        reconstructed_sequence = torch.cat(reconstructed_sequence[::-1])
        
        return reconstructed_sequence, lstm_out_hidden
    
    def forward(self, x, Vg, Jg, tick):
    #    if tick > 6:
    #        self.embedding.requires_grad = False
        Vg_emb = self.embedding_Vg(Vg).unsqueeze(0)
        Jg_emb = self.embedding_Jg(Jg).unsqueeze(0)
        latent_space_Mu, latent_space_Sigma,  embedded_input = self.encode(x)
        
        latent_space_reparam = self.reparametrize(latent_space_Mu, latent_space_Sigma)
       # print(latent_space_reparam.shape, Vg_emb.shape, Jg_emb.shape)
        latent_space = torch.cat( (Vg_emb, latent_space_reparam, Jg_emb), dim=2 )  ## new dimention of latent space! 
        reconstructed_sequence, reconstruction_out_hidden = self.decode(len(x), latent_space)
        return latent_space, latent_space_Mu, latent_space_Sigma, reconstructed_sequence, embedded_input
        
    def init_hidden(self):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

        
##################################___ MODEL 4 ___#######################################################

class autoencoder_rnn2(nn.Module):
    """ RNN-based autoencoder-classifier. Here we use disjoint architecture effectively coding two 
        independent encoder models, than a bottlenech and two independent decoders. The key point is
        bottleneck where three latent spaces are created with three different affine transformation (Linear layers): 
        one for forward pass of a sequence, another for backwards pass and the third - for concatenated result of
        two RNN readings. At the start of decoder we use 'mixture' terms, where initial hidden layer for lstm-based
        reconstructors (sequence generators) should be of equal dimention as the letter embeddings. Here we again
        use affine transformation on the laten spaces but obtain smaller than needed vectors. This is done to increase
        the importance of the common latent space from which 'missing' part of final hidden states are generated 
        and than concatenated with forward/backward-pass derived parts. This common latent space plays a role of penultimate
        layer for the 'removable' classifier, branching from the bottlenech with independent loss function.
        """
    def __init__(self, embedding_dim, hidden_dim, latent_space_dim, 
                 common_latent_space_dim, mixture_size, embedding_dim_VJ):
        super(autoencoder_rnn2, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.hidden_in_f = self.init_hidden(1)
        self.hidden_in_r = self.init_hidden(1)
        self.latent_space_f_data = None
        self.latent_space_r_data = None
        self.common_latent_space_data = None


        self.embedding_Vg  = nn.Embedding(Vgenes_size, embedding_dim_VJ)
        self.embedding_Jg = nn.Embedding(Jgenes_size, embedding_dim_VJ)
            
        self.embedding  = nn.Embedding(alphabet_size, embedding_dim)
        self.lstm_in_f = nn.LSTM(embedding_dim, hidden_dim)
        self.lstm_in_r = nn.LSTM(embedding_dim, hidden_dim)
#### we will use an admixture system starting with small addition from the latent space combining two RNNs
#### two RNN hidden outputs (dim=128) -> Linear conjunction into common latent space (dim~128?)
#### two separate decoders, eating corresponding hidden outputs but concatenated with 2+2 neurons representing
#### embedded Vs/Js and another ~4 spawning from common latent through two independent Linear layers.
#### resulting input hidden vectors still should be equal to the dimention of embedded residue, but we will expand
#### this up to e.g 42 (google how sparce embedding affects nn training potential)
        
        self.common_latent_space = nn.Linear(hidden_dim*2, common_latent_space_dim)
        
        self.latent_space_f = nn.Linear(hidden_dim, latent_space_dim)
        self.latent_space_r = nn.Linear(hidden_dim, latent_space_dim)

        self.mixture_to_f = nn.Linear(latent_space_dim, mixture_size)
        self.mixture_to_r = nn.Linear(latent_space_dim, mixture_size)

        self.reconstructor_f = nn.LSTM(embedding_dim, embedding_dim)
        self.reconstructor_r = nn.LSTM(embedding_dim, embedding_dim)
        
#### we basically use two independent loss functions only uniting them in the last stepp with simple addition
#### another approach would have been to use outputed arrays to convert into common array and then restore 
#### letters using cosine simularity, but this might introduce too much complication
    
    def encode(self, x, batch_size):
        embedding = self.embedding(x)
        rev_indx = torch.range(len(x)-1,0,-1, dtype=torch.long)
        embedding_r = embedding[rev_indx,:]

        lstm_out, lstm_in_hidden_f = self.lstm_in_f(embedding.view(len(x),batch_size,-1), self.hidden_in_f)  ### here we translate all sequence at once! 
        lstm_out, lstm_in_hidden_r = self.lstm_in_f(embedding_r.view(len(x),batch_size,-1), self.hidden_in_r)  ### here we translate all sequence at once! 
        lstm_in_hidden = torch.cat( (lstm_in_hidden_f[0], lstm_in_hidden_r[0]), dim=2 )
        self.common_latent_space_data = F.dropout( F.tanh(self.common_latent_space(lstm_in_hidden)), 0.5)
        self.latent_space_f_data = self.latent_space_f(lstm_in_hidden_f[0])
        self.latent_space_r_data = self.latent_space_r(lstm_in_hidden_r[0])

        return self.latent_space_f_data, self.latent_space_r_data, embedding, embedding_r
    
    def decode(self, input_length, latent_space_f, latent_space_r, batch_size): 
        
        
        previous_letter_f = torch.LongTensor([all_letters.index('X') for x in range(batch_size)]) ### lets try to reversely reconstruct from end of sequence symbol 
        previous_letter_f = self.embedding(previous_letter_f).view(1,batch_size,-1)
        
        previous_letter_r = torch.LongTensor([all_letters.index('B') for x in range(batch_size)]) ### lets try to reversely reconstruct from end of sequence symbol 
        previous_letter_r = self.embedding(previous_letter_r).view(1,batch_size,-1)
        
        reconstructed_sequence_f = [previous_letter_f]
        reconstructed_sequence_r = [previous_letter_r]
        
        common_lat_sp_f_addmixture = self.mixture_to_f(self.common_latent_space_data).view(1,batch_size,-1)
        common_lat_sp_r_addmixture = self.mixture_to_f(self.common_latent_space_data).view(1,batch_size,-1)
        
        lstm_out_hidden_f = torch.cat( (latent_space_f, common_lat_sp_r_addmixture), dim=2)
        lstm_out_hidden_r = torch.cat( (latent_space_r, common_lat_sp_f_addmixture), dim=2)
        
        lstm_out_hidden_f = (lstm_out_hidden_f.view(1, batch_size, -1), lstm_out_hidden_f.view(1, batch_size, -1))
        lstm_out_hidden_r = (lstm_out_hidden_r.view(1, batch_size, -1), lstm_out_hidden_r.view(1, batch_size, -1))

       # print( lstm_out_hidden[0].shape )
        for i in range(input_length-2):  ### change this for generator later!
            current_out_f, lstm_out_hidden_f = self.reconstructor_f(previous_letter_f.view(1, batch_size, -1), lstm_out_hidden_f)
            current_out_r, lstm_out_hidden_r = self.reconstructor_r(previous_letter_r.view(1, batch_size, -1), lstm_out_hidden_r)

            reconstructed_sequence_f.append(current_out_f)
            previous_letter_f = current_out_f
            reconstructed_sequence_r.append(current_out_r)
            previous_letter_r = current_out_r
            
        last_letter_f = torch.LongTensor([all_letters.index('B') for x in range(batch_size)]) ### lets try to reversely reconstruct from end of sequence symbol 
        last_letter_f = self.embedding(last_letter_f).view(1,batch_size,-1)
        last_letter_r = torch.LongTensor([all_letters.index('X') for x in range(batch_size)]) ### lets try to reversely reconstruct from end of sequence symbol 
        last_letter_r = self.embedding(last_letter_r).view(1,batch_size,-1)
        
        reconstructed_sequence_f.append(last_letter_f)
        reconstructed_sequence_r.append(last_letter_r)
        
        reconstructed_sequence_f = torch.cat(reconstructed_sequence_f[::-1], dim=0)
        reconstructed_sequence_r = torch.cat(reconstructed_sequence_r[::-1], dim=0)
        
        return reconstructed_sequence_f, reconstructed_sequence_r
    
    def forward(self, x, Vg, Jg, tick, batch_size):
    #    if tick > 6:
    #        self.embedding.requires_grad = False
        Vg_emb = self.embedding_Vg(Vg).view(1, batch_size, -1)
        Jg_emb = self.embedding_Jg(Jg).view(1, batch_size, -1)
        latent_space_f, latent_space_r, embedded_f, embedded_r = self.encode(x, batch_size)
       # print(latent_space.shape, Vg_emb.shape, Jg_emb.shape)
        latent_space_f = torch.cat( (Vg_emb, latent_space_f, Jg_emb), dim=2 )  ## new dimention of latent space!
        latent_space_r = torch.cat( (Vg_emb, latent_space_r, Jg_emb), dim=2 )  ## new dimention of latent space! 

        reconstructed_sequence_f, reconstructed_sequence_r = self.decode(len(x), latent_space_f, latent_space_r, batch_size)
        return latent_space_f, latent_space_r, reconstructed_sequence_f, reconstructed_sequence_r, embedded_f, embedded_r
#### Overall such approach might give more 'muscle' to our model and, what's more important -
#### allow more information to be stored in latent space (here we basically get three latent spaces, but the most
#### interest represent the common one - potentially it can store some 'additional' information about what makes
#### sequences actually functional

    def init_hidden(self, batch_size):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, batch_size, self.hidden_dim),
                torch.zeros(1, batch_size, self.hidden_dim))


##################################___ MODEL 5 ___#######################################################


class autoencoder_rnn3(nn.Module):
    
    def __init__(self, alphabet, embedding_dim, hidden_dim, hidden_dim2, latent_space_dim, 
                 common_latent_space_dim, mixture_size, embedding_dim_VJ):
        super(autoencoder_rnn3, self).__init__()
        
        self.alphabet = list(alphabet)
        self.hidden_dim = hidden_dim
        self.hidden_in_f = self.init_hidden(1)
        self.hidden_in_r = self.init_hidden(1)
        self.latent_space_f_data = None
        self.latent_space_r_data = None
        self.common_latent_space_data = None


        self.embedding_Vg  = nn.Embedding(Vgenes_size, embedding_dim_VJ)
        self.embedding_Jg = nn.Embedding(Jgenes_size, embedding_dim_VJ)
            
        self.embedding_kmers  = nn.Embedding(len(alphabet), embedding_dim)
        self.embedding  = nn.Embedding(len(all_letters), hidden_dim2)

        self.lstm_in_f = nn.LSTM(embedding_dim, hidden_dim)
        self.lstm_in_r = nn.LSTM(embedding_dim, hidden_dim)
#### we will use an admixture system starting with small addition from the latent space combining two RNNs
#### two RNN hidden outputs (dim=128) -> Linear conjunction into common latent space (dim~128?)
#### two separate decoders, eating corresponding hidden outputs but concatenated with 2+2 neurons representing
#### embedded Vs/Js and another ~4 spawning from common latent through two independent Linear layers.
#### resulting input hidden vectors still should be equal to the dimention of embedded residue, but we will expand
#### this up to e.g 42 (google how sparce embedding affects nn training potential)
        
        self.common_latent_space = nn.Linear(hidden_dim*2, common_latent_space_dim)
        
        self.latent_space_f = nn.Linear(hidden_dim, latent_space_dim)
        self.latent_space_r = nn.Linear(hidden_dim, latent_space_dim)

        self.mixture_to_f = nn.Linear(common_latent_space_dim+5, mixture_size)
        self.mixture_to_r = nn.Linear(common_latent_space_dim+5, mixture_size)

        self.reconstructor_f = nn.LSTM(hidden_dim2, hidden_dim2)
        self.reconstructor_r = nn.LSTM(hidden_dim2, hidden_dim2)
        
#### we basically use two independent loss functions only uniting them in the last stepp with simple addition
#### another approach would have been to use outputed arrays to convert into common array and then restore 
#### letters using cosine simularity, but this might introduce too much complication
    
    def encode(self, x, batch_size):
        embedding = self.embedding_kmers(x)
       # embedding_letters = self.embedding(x)
        rev_indx = torch.range(len(x)-1,0,-1, dtype=torch.long)
        embedding_r = embedding[rev_indx,:]

        lstm_out, lstm_in_hidden_f = self.lstm_in_f(embedding.view(len(x),batch_size,-1), self.hidden_in_f)  ### here we translate all sequence at once! 
        lstm_out, lstm_in_hidden_r = self.lstm_in_f(embedding_r.view(len(x),batch_size,-1), self.hidden_in_r)  ### here we translate all sequence at once! 
        lstm_in_hidden = torch.cat( (lstm_in_hidden_f[0], lstm_in_hidden_r[0]), dim=2 )
        self.common_latent_space_data = F.dropout( F.tanh(self.common_latent_space(lstm_in_hidden)), 0.5)
        self.latent_space_f_data = self.latent_space_f(lstm_in_hidden_f[0])
        self.latent_space_r_data = self.latent_space_r(lstm_in_hidden_r[0])

        return self.latent_space_f_data, self.latent_space_r_data, embedding, embedding_r
    
    def decode(self, input_length, latent_space_f, latent_space_r, batch_size): 
        
        
        previous_letter_f = torch.LongTensor([all_letters.index('X'*1) for x in range(batch_size)]) ### lets try to reversely reconstruct from end of sequence symbol 
        previous_letter_f = self.embedding(previous_letter_f).view(1,batch_size,-1)
        
        previous_letter_r = torch.LongTensor([all_letters.index('B'*1) for x in range(batch_size)]) ### lets try to reversely reconstruct from end of sequence symbol 
        previous_letter_r = self.embedding(previous_letter_r).view(1,batch_size,-1)
        
        reconstructed_sequence_f = [previous_letter_f]
        reconstructed_sequence_r = [previous_letter_r]
        
        common_lat_sp_f_addmixture = self.mixture_to_f(self.common_latent_space_data).view(1,batch_size,-1)
        common_lat_sp_r_addmixture = self.mixture_to_f(self.common_latent_space_data).view(1,batch_size,-1)
        
        lstm_out_hidden_f = torch.cat( (latent_space_f, common_lat_sp_r_addmixture), dim=2)
        lstm_out_hidden_r = torch.cat( (latent_space_r, common_lat_sp_f_addmixture), dim=2)
        
        lstm_out_hidden_f = (lstm_out_hidden_f.view(1, batch_size, -1), lstm_out_hidden_f.view(1, batch_size, -1))
        lstm_out_hidden_r = (lstm_out_hidden_r.view(1, batch_size, -1), lstm_out_hidden_r.view(1, batch_size, -1))

        #print(previous_letter_f.shape, lstm_out_hidden_f[0].shape, lstm_out_hidden_f[1].shape )
        for i in range(input_length-1):  ### change this for generator later!
            current_out_f, lstm_out_hidden_f = self.reconstructor_f(previous_letter_f.view(1, batch_size, -1), lstm_out_hidden_f)
            current_out_r, lstm_out_hidden_r = self.reconstructor_r(previous_letter_r.view(1, batch_size, -1), lstm_out_hidden_r)
            reconstructed_sequence_f.append(current_out_f)
            previous_letter_f = current_out_f
            reconstructed_sequence_r.append(current_out_r)
            previous_letter_r = current_out_r
            
        last_letter_f = torch.LongTensor([all_letters.index('B'*1) for x in range(batch_size)]) ### lets try to reversely reconstruct from end of sequence symbol 
        last_letter_f = self.embedding(last_letter_f).view(1,batch_size,-1)
        last_letter_r = torch.LongTensor([all_letters.index('X'*1) for x in range(batch_size)]) ### lets try to reversely reconstruct from end of sequence symbol 
        last_letter_r = self.embedding(last_letter_r).view(1,batch_size,-1)
        
        reconstructed_sequence_f.append(last_letter_f)
        reconstructed_sequence_r.append(last_letter_r)
        
        reconstructed_sequence_f = torch.cat(reconstructed_sequence_f[::-1], dim=0)
        reconstructed_sequence_r = torch.cat(reconstructed_sequence_r[::-1], dim=0)
        
        return reconstructed_sequence_f, reconstructed_sequence_r
    
    def forward(self, x, Vg, Jg, len_, batch_size):
    #    if tick > 6:
    #        self.embedding.requires_grad = False
        len_ = torch.Tensor([len_]*batch_size).view(1, batch_size, -1)
        Vg_emb = self.embedding_Vg(Vg).view(1, batch_size, -1)
        Jg_emb = self.embedding_Jg(Jg).view(1, batch_size, -1)
        latent_space_f, latent_space_r, embedded_f, embedded_r = self.encode(x, batch_size)
       # print(latent_space.shape, Vg_emb.shape, Jg_emb.shape)
        self.common_latent_space_data = torch.cat( (Vg_emb, self.common_latent_space_data, Jg_emb, len_), dim=2 )  ## new dimention of latent space!

        reconstructed_sequence_f, reconstructed_sequence_r = self.decode(len(x), latent_space_f, latent_space_r, batch_size)
        return latent_space_f, latent_space_r, reconstructed_sequence_f, reconstructed_sequence_r, embedded_f, embedded_r
#### Overall such approach might give more 'muscle' to our model and, what's more important -
#### allow more information to be stored in latent space (here we basically get three latent spaces, but the most
#### interest represent the common one - potentially it can store some 'additional' information about what makes
#### sequences actually functional

    def init_hidden(self, batch_size):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, batch_size, self.hidden_dim),
                torch.zeros(1, batch_size, self.hidden_dim))

        
    
##################################___ MODEL 6 ___#######################################################
        
class shallow_cnn_classifier(nn.Module):
    """ This model architecture is inpired by the success of CNN classifiers in the task of sentence
        class prediction (either by style/by mood/or by topic). Here we use a simplest model with single
        convolutional layer and max1-pooling from resulting feature maps to obtain independence of the 
        sequence length. Immideate concatenation of the resulting maximums is directly used by the classifier.
        """
    def __init__(self, embedding_dim, hidden_dim, latent_space_dim, 
                 common_latent_space_dim, mixture_size, embedding_dim_VJ):
        super(autoencoder_rnn2, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.hidden_in_f = self.init_hidden(1)
        self.hidden_in_r = self.init_hidden(1)
        self.latent_space_f_data = None
        self.latent_space_r_data = None
        self.common_latent_space_data = None        
    
    
    