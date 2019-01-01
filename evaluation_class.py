#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 16:49:07 2018

@author: lunar
"""

from training_class import *
from imports import *
from model_utils import *

class evaluation_results():
    def __init__(self):
        self.latent_reps = []
        self.core_seq_parts = []
        self.alignment_scores = []
        self.targets = []
        self.predicts = []


class evaluation_class(Training_loop):

    def __init__(self, aligner, prepare_batch, classifier=False, is_batched=False, return_seqs=True, hat_classifier=None):
        self.classifier = classifier
        self.batched = is_batched
        self.prepare_batch = prepare_batch
        self.return_seqs = return_seqs
        self.classifier_forward_sep = hat_classifier
        self.results = evaluation_results()

        self.aligner = aligner

    def __call__(self, model, data_loader, testing_set=False, strip=0):
        ### we flush all results after calling
        self.results = evaluation_results()

        if isinstance(testing_set, type(False)):
            testing_set = data_loader.test
        with torch.no_grad():
            if self.batched:
                ## faster but does not allow to assign results to the sequences easily
                indices = []
                for len_ in testing_set['Len'].unique():
                    batch = testing_set[testing_set['Len']==len_]
                    batch_size = batch.shape[0]

                    model.zero_grad()
                    model.hidden_in_f = model.init_hidden(batch_size)
                    model.hidden_in_r = model.init_hidden(batch_size)

                    self.prepare_batch(batch, data_loader, strip=strip)

                    batch_target = batch['Pred'].values.tolist()
                    self.results.targets += batch_target
                    batch_target = torch.Tensor(batch_target)
                    indices += list(batch.index)

                    latent_space_f, latent_space_r, rec_seq_emb_f, rec_seq_emb_r, emb_input, emb_input_r = model(self.prepare_batch.seq, self.prepare_batch.v, self.prepare_batch.j, 1, batch_size)
                    self.results.latent_reps.append( list(map(lambda x: x.numpy().reshape(batch_size, -1), ( latent_space_f, latent_space_r, model.common_latent_space_data )) ))

                    if self.classifier:
                        classifier_results = self.classifier_forward_sep(model, batch_size)
                        #predict = torch.argmax(classifier_results, dim=-1).numpy().tolist()
                        predict = classifier_results.numpy().tolist()
                        self.results.predicts += predict

                    if self.return_seqs:
                        fully_reconstructed_seqs_f, rec_ds_f = reconstruct_output_sequence(model, rec_seq_emb_f, data_loader.all_letters, '', batch_size)
                        fully_reconstructed_seqs_r, rec_ds_r = reconstruct_output_sequence(model, rec_seq_emb_r, data_loader.all_letters, '', batch_size)
                        self.results.core_seq_parts.append(( batch['seq'].values, fully_reconstructed_seqs_f, fully_reconstructed_seqs_r[0]))

                        a1 = self.alignment_one(fully_reconstructed_seqs_r, batch, divisor=1, reverse=False, return_all=True)
                        a2 = self.alignment_one(fully_reconstructed_seqs_f, batch, divisor=1, reverse=True, return_all=True)
                        self.results.alignment_scores += [[a1,a2]]
            elif not self.batched:
                ## slower, but returns results in the predictable order
                batch_size = 1
                for ind in range( testing_set.shape[0] ):
                    w = testing_set.iloc[ind,:]
                    self.results.targets.append(w[3])
                    input = prepare_sequence(read_kmers_with_spacing(w[0], data_loader.kmers_n, 1), data_loader.kmer_to_ix)
                    input_r = prepare_sequence(read_kmers_with_spacing(w[0][::-1], data_loader.kmers_n, 1), data_loader.kmer_to_ix)
                    len_ = len(w[0]) / 50
                    Vg = prepare_sequence([w[1]], data_loader.V_to_ix)
                    Jg = prepare_sequence([w[2]], data_loader.J_to_ix)

                    model.hidden_in_f = model.init_hidden(1)  #### Dont forget to init hiddens,  idiot!
                    model.hidden_in_r = model.init_hidden(1)
                    ########### model run
                    latent_space_f, latent_space_r, rec_seq_emb_f, rec_seq_emb_r, emb_input, emb_input_r = model(input, Vg, Jg, len_, 1)
                    self.results.latent_reps.append( list(map(lambda x: x.numpy().reshape(1,-1), ( latent_space_f, latent_space_r, model.common_latent_space_data )) ))

                    if self.classifier:
                        classifier_results = self.classifier_forward_sep(model, batch_size)
                        #predict = torch.argmax(classifier_results, dim=-1).numpy().tolist()
                        predict = classifier_results.numpy().tolist()
                        self.results.predicts += predict

                    if self.return_seqs:
                        fully_reconstructed_seqs_f, rec_ds_f = reconstruct_output_sequence(model, rec_seq_emb_f, data_loader.all_letters, '', batch_size)
                        fully_reconstructed_seqs_r, rec_ds_r = reconstruct_output_sequence(model, rec_seq_emb_r, data_loader.all_letters, '', batch_size)
                        self.results.core_seq_parts.append((w[0][3:-3], fully_reconstructed_seqs_f[0][3:-3], fully_reconstructed_seqs_r[0][3:-3]))

                        self.results.alignment_scores.append(

                            (self.aligner.align(w[0][1:-1], fully_reconstructed_seqs_f[0][1:-1])[0].score,
                             self.aligner.align(w[0][::-1][1:-1], fully_reconstructed_seqs_r[0][1:-1])[0].score))
    def get_numpy(self):
        self.results.predicts = np.array(self.results.predicts)
        self.results.targets = np.array(self.results.targets)
        self.results.core_seq_parts = np.array(self.results.core_seq_parts)
        self.results.alignment_scores = np.array(self.results.alignment_scores)
        tmp = []
        tmp.append( np.array([i[0] for i in self.results.latent_reps]) )
        tmp.append( np.array([i[1] for i in self.results.latent_reps]) )
        tmp.append( np.array([i[2] for i in self.results.latent_reps]) )
        self.results.latent_reps = tmp



def evaluate(model, set_, char_to_ix, batch_size=1, return_seqs = False):
    alignment_scores = []
    core_sequence_parts = []
    latent_reps_eval = []
    p = []
    targets = []
    #random_sample  = np.random.choice( np.arange(0, len(set_)), len(set_), replace=False)
    for ind in range(len(set_)): #random_sample:
        with torch.no_grad():
            ######### prepare input with batch ZERO
            w = set_[ind]
            targets.append(w[-1])
            input = prepare_sequence(read_kmers_with_spacing(w[0],1), char_to_ix )
            
            input_r = prepare_sequence(read_kmers_with_spacing(w[0][::-1],1), char_to_ix)
            len_ = len(w[0])/50
            Vg = prepare_sequence([w[1]], V_to_ix)
            Jg = prepare_sequence([w[2]], J_to_ix)    
            ######### zero hiddens
            model.hidden_in_f = model.init_hidden(batch_size)  #### Dont forget to init hiddens,  idiot!
            model.hidden_in_r = model.init_hidden(batch_size)
            ########### model run
            lat_f, lat_r, rec_seq_emb_f, rec_seq_emb_r, emb_inp, emb_inp_r = model(input, Vg, Jg, len_, batch_size)
            latent_reps_eval.append(model.common_latent_space_data.numpy().reshape(-1))
#           latent_reps_eval.append( np.concatenate( (model.common_latent_space_data.numpy().reshape(-1), 
#                                            lat_f.numpy().reshape(-1), lat_r.numpy().reshape(-1),)))
#             ########### classification
#             classifier_results = classifier_forward_sep(model, batch_size)
#             predict = torch.argmax(classifier_results, dim=1).numpy().tolist()
#             p += predict
            ################# for sequense reconstructions
            if return_seqs:
                fully_reconstructed_seqs_f, rec_ds_f = reconstruct_output_sequence(rec_seq_emb_f, all_letters, '', batch_size)
                fully_reconstructed_seqs_r, rec_ds_r = reconstruct_output_sequence(rec_seq_emb_r, all_letters, '', batch_size)
                core_sequence_parts.append( (w[0][3:-3], fully_reconstructed_seqs_f[0][3:-3], fully_reconstructed_seqs_r[0][3:-3] ) )
                alignment_scores.append( ( aligner.align(w[0][1:-1], fully_reconstructed_seqs_f[0][1:-1])[0].score,
                                           aligner.align(w[0][::-1][1:-1], fully_reconstructed_seqs_r[0][1:-1])[0].score ) )
    targets = np.array(targets)
    p = np.array(p)  
    
    return targets, p, alignment_scores, core_sequence_parts, np.array(latent_reps_eval) #.mean(axis=0), np.array(latent_reps_eval).std(axis=0)