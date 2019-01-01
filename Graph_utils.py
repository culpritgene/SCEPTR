


import igraph
import numpy as np


def make_graph(ds_ki, tr):
    ki = (ds_ki > tr).astype('int').values
    np.fill_diagonal( ki, 0)
    ki_graph = igraph.Graph.Adjacency( ki.tolist(), mode=igraph.ADJ_UNDIRECTED)
    return ki_graph

def add_metadata(g, train_df, donor_coloring, yf_coloring, ):
    for i,v in enumerate(g.vs):
        v['donor']=donor_coloring[i]
        v['YF']=yf_coloring[i]
        v['seq']=train_df['seq'].values[i]
        v['gen_p']=train_df['gen_p'].values[i]

def make_induced_filtered(g, f=1):
    ### at least three sequences in cluster
    multitones_ki = []
    for v in g.vs:
        if v.degree() > f:
            multitones_ki.append(v.index)
    return multitones_ki

def del_zero_pgen(g):
    ### delete zero_pgen nodes
    zero_pgen = [v for v in  g.vs if v['gen_p']==0]
    g.delete_vertices(zero_pgen)

def count_degrees(g):
    degrees_ki = []
    for v in g.vs:
        degrees_ki.append(v.degree())
    return degrees_ki

def color_graph(g, palette, by='donor'):
    ### color induced_subgraph
    for v in g.vs:
        v['color']=palette[v[by]]