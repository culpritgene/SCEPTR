#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 11:28:25 2018

@author: lunar
"""
from imports import *
from model_utils import *
from data_loader import *
from evaluation_class import *
from training_class import *
from All_models import *

path_0 = '/home/lunar/Desktop/Potential_Diploma/'
path ='/home/lunar/Desktop/Potential_Diploma/classifier_data/'
donors = ['Azh', 'Luci', 'Kar', 'KB', 'GS', 'Yzh']
filename_mask = lambda x: f"leave_one_out/{x}_one_out.txt"
all_letters = "BCQWERTYIPASDGHKLMNVFX"

my_data_loader = data_loader(path=path, type_='donors', kmers_n=3, letters = None )
my_data_loader(donors=donors, filename_mask=filename_mask, filter_= lambda x: 1)

all_Vgenes = my_data_loader.all_Vgenes
all_Jgenes = my_data_loader.all_Jgenes
V_to_ix = my_data_loader.V_to_ix
J_to_ix = my_data_loader.J_to_ix
char_to_ix = my_data_loader.char_to_ix
kmer_to_ix = my_data_loader.kmer_to_ix
all_kmers = my_data_loader.all_kmers

#print(kmer_to_ix)

### decoder hidden_dim should be equal to embedding dim (e.g 42)
### we concatenate 2+2 neurons representing V/J segments, and another 6 from common space infusion
### thus initial proto-'hidden_dim' should be 42-10=32
model = autoencoder_rnn3(alphabet= all_kmers,
                         Vgenes_size=len(all_Vgenes),
                         Jgenes_size=len(all_Jgenes),
                         all_letters=all_letters,
                         embedding_dim=256, # 64
                         hidden_dim=256,
                         hidden_dim2=128, #### assert hidden_dim2==embedding_dim
                         latent_space_dim=64,
                         common_latent_space_dim=128,
                         mixture_size=64,
                         embedding_dim_VJ=2,)
### 48 = 30 + 14 + 2*2
### 42 = 16 + 2 + 4
optimizer = optim.Adam( filter(lambda p: p.requires_grad , model.parameters()), lr=0.001)

sub_matrix = pickle.load( open(path_0 +'Supplementary/Substitution_table_kidera_default.json', 'br' ) )

sub_matrix = {k:v/16 for k,v in sub_matrix.items()}

aligner = Align.PairwiseAligner()
aligner.substitution_matrix = sub_matrix
aligner.end_open_gap_score = -1
aligner.end_extend_gap_score = -1
aligner.internal_open_gap_score = -7
aligner.internal_extend_gap_score = -4


evaluate = evaluation_class(aligner)

prepare_batch_ =  prepare_batch()

training_loop = Training_loop(aligner, optimizer, prepare_batch=prepare_batch_, reconstruct_output_sequence=reconstruct_output_sequence, path_to_checkpoints=path_0+'Model_checkpoints/',  model_name='AE_letters_fixed_margins', evaluate=False, run_name='default_AE_training')

training_loop.run_loop(model, data_loader=my_data_loader, epochs=250, selected_batch_size=12, strip=0, voice=50)


save_checkpoint({
            'epoch': 250,
            'state_dict': model.state_dict(),
            'all_features': training_loop.stats,
            'optimizer' : optimizer.state_dict(),
        },  'Yes',
            'checkpoint_bidirectional_mixture_AE_V10_Large_KMERS3_KMERS1__BatchNorm_Reдыlu_250.pth.tar')