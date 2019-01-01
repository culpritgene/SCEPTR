#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 16:42:20 2018

@author: lunar
"""

from imports import *
from model_utils import *

class data_loader():
    def __init__(self, path, type_, kmers_n, letters=None, donors=None):
        self.type = type_
        self.kmers_n = kmers_n
        self.path = path
        self.donors = {}
        if letters==None:
            self.all_letters = "BCQWERTYIPASDGHKLMNVFX"
        else:
            self.all_letters = letters
        self.all_kmers = []
        self.all_Vgenes = None
        self.all_Jgenes = None
        self.V_to_ix = None
        self.J_to_ix = None
        self.char_to_ix = None
        self.kmer_to_ix = None
        self.data = None
        self.train = None
        self.test = None

    def __call__(self,  donors=None, filename_mask=None, filter_= lambda x: 1):
        if self.type=='donors':
            for donor in donors:
                path_to_donor_data = self.path + filename_mask(donor)
                data = pd.read_csv(path_to_donor_data, sep='\t')
                data.columns =data.columns.str.replace("\.", "_") ### remove R styled columns notations
                data['check_for_stopcodon'] = data['CDR3_amino_acid_sequence'].apply(check_for_asterics) ## basic filter for stop_codons in CDR3
                data['check_for_unknown'] = data['CDR3_amino_acid_sequence'].apply(check_for_unknown)  ## basic filter for sequence uncertainty
                data['filter'] =  data.apply(filter_)
                data['Subj'] = donor
                data = data[(data['check_for_unknown'] == 0) & (data['check_for_stopcodon'] == 0)]
                self.donors.update({donor:data})
            self.data = pd.concat( list(self.donors.values()) )
            self.data.index = np.arange( self.data.shape[0] )
        elif self.type=='all':
            data = pd.read_csv(self.path)
            data.columns = data.columns.str.replace("\.", "_")
            self.data = data

        try:
            data.columns = data.columns.str.replace("\.", "_")  ### remove R styled columns notations
            data['check_for_stopcodon'] = data['CDR3_amino_acid_sequence'].apply( check_for_asterics)  ## basic filter for stop_codons in CDR3
            data['check_for_unknown'] = data['CDR3_amino_acid_sequence'].apply(check_for_unknown)  ## basic filter for sequence uncertainty
            data['filter'] = data.apply(filter_)
            data = data[(data['check_for_unknown'] == 0) & (data['check_for_stopcodon'] == 0)]
        except KeyError:
            pass

        #### encode V and J genes as well as single residues/extra characters
        self.encode_VJ()
        self.char_to_ix = {ac : ind for ind, ac in enumerate(self.all_letters)}
        
        ### make kmers if needed
        if self.kmers_n > 1:
            for i in itertools.product(self.all_letters, repeat=self.kmers_n):
                self.all_kmers.append((''.join(i)))
            self.all_kmers = np.array(self.all_kmers)
            self.kmer_to_ix = {ac : ind for ind, ac in enumerate(self.all_kmers)}
        
        ### split data into train and test
        self.train = self.data[['CDR3_amino_acid_sequence', 'bestVGene', 'bestJGene', 'YF', 'Subj']]
        self.train.columns = ['seq', 'V', 'J', 'Pred', 'Subj']
        self.train['Len'] = self.train['seq'].apply(len)
        self.train['seq'] = self.train['seq'].apply(lambda x: 'B'+x+'X')
        self.train['Pred'] = self.train['Pred'].astype('int')

        ### globalize useful varaibles
        self.globalize()



    def globalize(self, ):
        global all_Vgenes
        global all_Jgenes
        global V_to_iv
        global J_to_ix
        global char_to_ix
        global kmer_to_ix

        all_Vgenes = self.all_Vgenes
        all_Jgenes = self.all_Jgenes
        V_to_ix = self.V_to_ix
        J_to_ix = self.J_to_ix
        char_to_ix = self.char_to_ix
        kmer_to_ix = self.kmer_to_ix


    def encode_VJ(self, ):
        self.all_Vgenes = self.data['bestVGene'].unique()
        self.all_Jgenes = self.data['bestJGene'].unique()
        self.V_to_ix = {ac: ind for ind, ac in enumerate(self.all_Vgenes)}
        self.J_to_ix = {ac: ind for ind, ac in enumerate(self.all_Jgenes)}

    def update(self, new_df_part):
        self.data = pd.concatenate([self.data, new_df_part])
        self.encode_VJ()
        self.globalize()
