#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 16:41:56 2018

@author: lunar
"""

from imports import *
from model_utils import *

class training_stats():
     def __init__(self, thrsh):
        self.all_predicts = []
        self.all_targets = []
        self.all_losses = []
        self.all_losses_classifier = []
        self.all_alignment_scores_f = []
        self.all_alignment_scores_r = []
        self.all_accs = []
        self.all_precisions = []
        self.all_recalls = []
        self.VAE_losses = []
        ########## test
        self.test_results = [] ## targets, predicts, acc, precision, recall
        self.latent_test_mean = []
        self.latent_test_std = []

        self.best_test_accuracy  = thrsh


class Loss():
    def __init__(self, mode='history'):
        losses_stack = ['total_loss', 'classifier_loss', 'loss_f', 'loss_r', 'KLD_loss', 'regularized_loss']
        assert mode in ['history', 'epoch', 'batch'], print('Unknown type of mode for Loss storage')

        if mode == 'history' or 'epoch':
            for ls in losses_stack:
                setattr(self, ls, [])
        if mode == 'batch':
            for ls in losses_stack:
                setattr(self, ls, 0)

    def take_mean(self):
        for k,v in self.__dict__.items():
            setattr(self, k, np.mean(v))



class Training_loop_coeffs():
    def __init__(self):
        self.VAE_coeff = 0.01
        self.aligment_f_c = 1
        self.aligment_r_c = 1
        self.weight_1 = 0.72
        self.weight_0 = 1 - self.weight_1
        self.thrsh_1 = 0.5

class Training_loop():
    def __init__(self, aligner, optimizer, prepare_batch, reconstruct_output_sequence, model_name, run_name, path_to_checkpoints, classifier_mode=False, evaluate_mode=False, evaluate=False,  VAE_mode=False, thrsh=0.55):
               
        self.stats = training_stats(thrsh)
        self.History_loss = Loss(mode='history')
        self.Epoch_loss = Loss(mode='epoch')
        self.Batch_loss = Loss(mode='batch')

        self.parameters = Training_loop_coeffs()

        self.model_name = model_name
        self.prepare_batch = prepare_batch
        self.reconstruct_output_sequence = reconstruct_output_sequence
        self.run_name = run_name
        self.classifier_forward_sep = None
        self.classifier_mode = classifier_mode
        self.VAE_mode = VAE_mode
        self.evaluate_mode = evaluate_mode
        self.evaluate = evaluate
        self.pt = path_to_checkpoints

        self.regularize = False
        self.aligner = aligner
        self.optimizer = optimizer
        
        self.batch = None
        self.batch_seq, self.batch_seq_r, self.batch_v, self.batch_j, = None, None, None, None
        self.batch_target = None


        self.latent_space_f = None
        self.latent_space_r = None
        self.rec_seq_emb_f = None
        self.rec_seq_emb_r = None
        self.emb_input = None
        self.emb_input_r = None
        self.rec_ds_f = None
        self.rec_ds_r = None

        self.current_e = 0

    def extra_functions(self):
        pass

    def drop_to_legacy_e(self, new_e):
        ans = input(f'Are you sure that you want to delete all stats of epochs above {new_e} (y/n)')
        if ans.lower() == 'y' or 'yes':

            print(f'Setting current epoch to {new_e}... \n deleting all above stats and history losses.')
            ### For stats
            self.cut_above_losses(self.stats, new_e)
            ### Same for History Losses
            self.cut_above_losses(self.History_loss, new_e)

            self.current_e = new_e

        elif ans.lower() == 'n' or 'no':
            print('Closing!')
        else:
            print('Your answer should be either Yes (y) or No (n).')

    def cut_above_losses(self, inner_cls, new_e):
        for k, v in inner_cls.__dict__.items():
            try:
                if len(v) == self.current_e:
                    print(k)
                    setattr(inner_cls, k, v[:new_e])
            except TypeError:
                pass


    def loss_function_KLD(self, mu, logvar):
        KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        KLD = torch.sum(KLD_element).mul_(-0.5)
        return KLD

    def criterion(self, model, batch, batch_seq, batch_seq_r,):
         self.Batch_loss.loss_f = F.nll_loss(F.log_softmax(self.rec_ds_f, dim=2).permute(0,2,1), batch_seq.permute(1,0))
         self.Batch_loss.loss_r = F.nll_loss(F.log_softmax(self.rec_ds_r, dim=2).permute(0,2,1), batch_seq_r.permute(1,0))
         
         self.alignment_score_f = self.alignment_one(self.fully_reconstructed_seqs_f, batch)
         self.alignment_score_r = self.alignment_one(self.fully_reconstructed_seqs_r, batch, reverse=True)
         self.Batch_loss.total_loss = self.Batch_loss.loss_f*self.alignment_score_f + self.Batch_loss.loss_r*self.alignment_score_r

         if self.classifier_mode:
             self.Batch_loss.classifier_loss = F.binary_cross_entropy(self.classifier_results, self.batch_target, weight=torch.FloatTensor([self.parameters.weight_1]))
             self.Batch_loss.total_loss += self.Batch_loss.classifier_loss

         if self.VAE_mode:
             self.Batch_loss.KLD_loss = self.loss_function_KLD(model.common_latent_mu, model.common_latent_sigma)
             self.Batch_loss.total_loss += self.Batch_loss.KLD_loss*self.parameters.VAE_coeff

         if self.regularize:
             self.Batch_loss.regularized_loss = self.regularize(model)
             self.Batch_loss.total_loss += self.Batch_loss.regularized_loss

         #return self.Batch_loss.total_loss
         
    def alignment_one(self, reconstructions, batch, reverse=False, return_all=False):

        divisor = self.parameters.aligment_f_c
        if reverse:
            divisor = self.parameters.aligment_r_c

        alignment_score = []
        for i, xx in enumerate(reconstructions):
            try:
                if reverse:
                    xx = xx[::-1]
                alignment_score.append( self.aligner.align(xx[1:-1], list(batch['seq'])[i][1:-1])[0].score )
            except ValueError:
                pass
        if return_all:
            return alignment_score
        else:
            alignment_score = -np.mean(alignment_score) / divisor
            if (alignment_score < -400 or alignment_score > 400):
                alignment_score = 1
            return alignment_score
    
    def verbolize(self, e, batch, len_, strip):
        print(f'Epoch: {e}, batch shape: {batch.shape}, loss: {round(self.Batch_loss.total_loss.item(),4)}')
        if self.VAE_mode:
            print(f'VAE KLD Loss: {self.Batch_loss.KLD_loss}')
        print(f'loss_f: {round(self.Batch_loss.loss_f.item(),3)}, loss_r: {round(self.Batch_loss.loss_f.item(),3)}, alignment_f: {round(self.alignment_score_f,3)}, alignment_r: {round(self.alignment_score_r,3)}')
        print('direct:............')
        print('Original seqsin:', [x[strip:len_-strip] for x in batch['seq'].tolist()[:3]])
        print('Reconstructions:', self.fully_reconstructed_seqs_f[:3])
        print('reverse:............')
        print('Original seqsin:', [x[strip:len_-strip][::-1] for x in batch['seq'].tolist()[:3]])
        print('Reconstructions:', self.fully_reconstructed_seqs_r[:3])
    
    def verbolize_test(self,e,s):
        print(f'Epoch {e}, TRAIN Accuracy {round(self.stats.all_accs[-1],3)}, Precision {round(self.stats.all_precisions[-1],3)}, Recall {round(self.stats.all_recalls[-1],3)}')
        print(f'Epoch {e}, TEST  Accuracy {round(s[0],3)}, Precision {round(s[1],3)}, Recall {round(s[2],3)}')
    
    def append_train_classifier_stats(self, targets, predicts):
        """ This function is used only in the classifier mode"""
        #self.stats.all_predicts.append(predicts)
        #self.stats.all_targets.append(targets)
        self.stats.all_accs.append(accuracy_score(targets, predicts) )
        self.stats.all_precisions.append( precision_score( targets, predicts ))
        self.stats.all_recalls.append( recall_score( targets, predicts ))

    def run_loop(self, model,  data_loader , epochs, test_donor=None, selected_batch_size=12, strip=0, voice=50, classifier_mode=False):
        ####### make from this a generator!!
        for e in range(epochs):
            total_loss = []
            total_alignment_score_f = []
            total_alignment_score_r = []

            del self.Epoch_loss
            self.Epoch_loss = Loss(mode='epoch')


            train = data_loader.train
            train_df_2 = train[train['Subj']!=test_donor].copy() ### mask testing donor
            testing_set = train[train['Subj']==test_donor]
            ITER = 0
            predicts = []
            targets = []
            while train_df_2.shape[0] > 1:
                ### universal randomized exhausting sampling from training database
                train_df_2, batch = take_batch_by_len(train_df_2, batch_length=selected_batch_size)
                batch_size = batch.shape[0] # sometimes batch shapes are less than stated
                ### preparation of batch data - V,J,Len and other non-CDR3 parameters encoding, CDR3 encoding by residues or kmers
                self.prepare_batch(batch, data_loader, strip=strip) ### we save prepared batch in a special class
                ### here batch target is taken - to draw classification metrics on training epoch
                self.batch_target = torch.Tensor(batch['Pred'].values.tolist())
        
                model.zero_grad()
               # model.hidden_in_r = model.init_hidden(batch_size, hidden_type='kmer') ### Dont forget to init hiddens,  you fool!
               # model.hidden_in_r = model.init_hidden(batch_size, hidden_type='kmer') ### Dont forget to init hiddens,  you fool!
                model.hidden_char_f = model.init_hidden(1, hidden_type='char') ### Dont forget to init hiddens,  you fool!
                model.hidden_char_r = model.init_hidden(1, hidden_type='char') ### Dont forget to init hiddens,  you fool!
                model.hidden_kmer_f = model.init_hidden(1, hidden_type='kmer')
                model.hidden_kmer_r = model.init_hidden(1, hidden_type='kmer')

        
                self.latent_space_f, self.latent_space_r, self.rec_seq_emb_f, self.rec_seq_emb_r, self.emb_input, self.emb_input_r = model(self.prepare_batch.model_input, self.prepare_batch.v, self.prepare_batch.j ,1, batch_size)
                self.fully_reconstructed_seqs_f, self.rec_ds_f = self.reconstruct_output_sequence(model, self.rec_seq_emb_f, data_loader.all_letters, '', batch_size)
                self.fully_reconstructed_seqs_r, self.rec_ds_r = self.reconstruct_output_sequence(model, self.rec_seq_emb_r, data_loader.all_letters, '', batch_size)

                if self.classifier_mode:
                    self.classifier_results = self.classifier_forward_sep(model, batch_size)
                    #predict = torch.argmax(self.classifier_results, dim=1).numpy()

                    ### set classifier results from probabilities to booleans so accuracy/ets will work
                    ### this doesn't affect training in any way
                    self.classifier_predicts = self.classifier_results > self.parameters.thrsh_1
                    targets += self.batch_target.numpy().tolist()
                    predicts += self.classifier_predicts.numpy().tolist()

                ### Compute All Batch Losses
                self.criterion(model, batch, self.prepare_batch.seq, self.prepare_batch.seq_r, )

                ### Append All Batch Loss parameters to current Epoch Loss
                for k,v in self.Batch_loss.__dict__.items():
                    try:
                        v = v.item()
                    except AttributeError:
                        pass
                    self.Epoch_loss.__dict__[k].append(v)

                total_alignment_score_f.append( self.alignment_score_f)
                total_alignment_score_r.append( self.alignment_score_r)

                self.Batch_loss.total_loss.backward()
                self.optimizer.step()

                ITER += 1
                if ITER % voice == 0:
                    self.verbolize(e, batch, self.prepare_batch.len_, strip)

                self.extra_functions()

            ### End of Epoch - writing down Stats of Epoch and Evaluation
            self.stats.current_e = e
            self.Epoch_loss.take_mean()
            for k, v in self.Epoch_loss.__dict__.items():
                self.History_loss.__dict__[k].append(v)

            ### All losses are now in History_loss object, not in stats!
            self.stats.all_losses.append( self.Epoch_loss.total_loss )
            self.stats.all_alignment_scores_f.append( np.mean(total_alignment_score_f))
            self.stats.all_alignment_scores_r.append( np.mean(total_alignment_score_r))


            if self.evaluate_mode:
                self.evaluate(model, data_loader, testing_set, )
                self.evaluate.get_numpy()
                self.stats.test_results.append( self.evaluate.results )

            # if self.VAE_mode:
            #     self.stats.VAE_losses.append( np.mean(vae_loss) )


            if self.classifier_mode:
                self.append_train_classifier_stats(targets, predicts)
                test_targets = self.evaluate.results.targets
                test_p = self.evaluate.results.predicts > self.parameters.thrsh_1
                s = [accuracy_score(test_targets, test_p), precision_score(test_targets, test_p), recall_score(test_targets, test_p)]
                self.verbolize_test(e,s)

