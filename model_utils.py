#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 16:41:53 2018

@author: lunar
"""
from imports import *


def check_for_asterics(x):
    if re.findall('\*', x):
        return 1
    else:
        return 0

def check_for_unknown(x):
    if re.findall('\_', x):
        return 1
    else:
        return 0

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.LongTensor(idxs)

def read_kmers_with_spacing(t, kn, sp):
    sentence = []
    for i in range(1,len(t),sp):
        seg = t[i - 1:i + kn - 1]
        if seg!='':
            if len(seg) != kn:
                seg = t[-kn:]
            sentence.append(seg)
    return sentence

def ix_to_char(ix, all_letters):
    word = ''
    for i in ix:
        word+=all_letters[i][0]
    return word

def cosine_distance_pytorch(a, b):
    a_norm = a / a.norm(dim=2).unsqueeze(-1)
    b_norm = b / b.norm(dim=1).unsqueeze(-1)
    res = torch.matmul(a_norm.permute(1,0,2), b_norm.transpose(0,1))
    return res

def take_batch_by_len(train_df_2, batch_length=32):
    len_ = np.random.choice(train_df_2['Len'].unique())
    sele = train_df_2[train_df_2['Len']==len_]
    if len(sele.index) < batch_length:
        batch_length = len(sele.index)
    sele_batch = np.random.choice(sele.index, batch_length, replace=False)
    sele_seqs = sele.loc[sele_batch]
    train_df_2 = train_df_2.drop(sele_batch)
    return train_df_2, sele_seqs


def reconstruct_output_sequence(model, rec_seq_emb, all_symbols, add, batch_size):
    rec_seq = cosine_distance_pytorch(rec_seq_emb, model.embedding.weight.data)
    assert rec_seq.shape[0] == batch_size, print('Dimention mismatch!')
    rec_seq_long = torch.argmax(rec_seq, dim=2)
    res = []
    for i in range(rec_seq.shape[0]):
        res.append( ix_to_char(rec_seq_long[i,:], all_symbols) +add )
    return res, rec_seq


class prepare_batch():
    def __init__(self):
        self.batch = None
        self.seq = None
        self.seq_r = None
        self.kmer_seq = None
        self.kmer_seq_r = None
        self.model_input = None
        self.v = None
        self.j = None
        self.classifier_targets  = None

    def __call__(self, batch, data_loader, strip=0):

        self.batch = batch
        self.len_ = len(batch['seq'].tolist()[0])
        self.seq = torch.cat(batch['seq'].apply(
            lambda x: prepare_sequence(x[strip:self.len_ - strip], data_loader.char_to_ix).unsqueeze(1)).values.tolist(), dim=1)
        self.seq_r = torch.cat(batch['seq'].apply(
            lambda x: prepare_sequence(x[strip:self.len_ - strip][::-1], data_loader.char_to_ix).unsqueeze(1)).values.tolist(), dim=1)

        self.model_input = self.seq

        if data_loader.kmer_to_ix:
            self.kmer_seq = torch.cat(batch['seq'].apply(lambda x: prepare_sequence(
                read_kmers_with_spacing(x[strip:self.len_ - strip], data_loader.kmers_n, 1), data_loader.kmer_to_ix).unsqueeze(1)).values.tolist(), dim=1)
            # kmer_seq_r = torch.cat( sele_seq['seq'].apply(lambda x: prepare_sequence(
            #     read_kmers_with_spacing( x[strip:len_-strip][::-1], data_loader.kmers_n, 1), kmer_to_ix).unsqueeze(1)).values.tolist(), dim=1)
            self.model_input = self.kmer_seq


        self.v = prepare_sequence(batch['V'].values, data_loader.V_to_ix).unsqueeze(1)
        self.j = prepare_sequence(batch['J'].values, data_loader.J_to_ix).unsqueeze(1)
        self.classifier_targets = torch.LongTensor(batch['Pred'].values.tolist())

def prepare_batch_kmer(sele_seq, kmer_to_ix, strip=0):
    len_ = len(sele_seq['seq'].tolist()[0])
    seq = torch.cat( sele_seq['seq'].apply(lambda x: prepare_sequence(x[strip:len_-strip], char_to_ix).unsqueeze(1)).values.tolist(), dim=1)
    seq_r = torch.cat( sele_seq['seq'].apply(lambda x: prepare_sequence(x[strip:len_-strip][::-1], char_to_ix).unsqueeze(1)).values.tolist(), dim=1)
    
    kmer_seq = torch.cat( sele_seq['seq'].apply(lambda x: prepare_sequence(
        read_kmers_with_spacing( x[strip:len_-strip], 1), kmer_to_ix).unsqueeze(1)).values.tolist(), dim=1)

    # kmer_seq_r = torch.cat( sele_seq['seq'].apply(lambda x: prepare_sequence(
    #     read_kmers_with_spacing( x[strip:len_-strip][::-1], 1), kmer_to_ix).unsqueeze(1)).values.tolist(), dim=1)

    v = prepare_sequence( sele_seq['V'].values , V_to_ix).unsqueeze(1)
    j = prepare_sequence( sele_seq['J'].values , J_to_ix).unsqueeze(1)
    classifier_targets = torch.LongTensor( sele_seq['Pred'].values.tolist())
    return kmer_seq, seq, seq_r, v, j, len_, classifier_targets

def save_checkpoint(state, is_best, filename='checkpoint_small_autoencoder.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')