#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 16:41:37 2018

@author: lunar
"""


import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import scipy
import shutil
import types
import itertools
import pickle

import seaborn as sns
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import xgboost as xgb
from sklearn.utils import shuffle
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc

import Bio
from sklearn import manifold, decomposition
from Bio import Align, pairwise2
from Bio.pairwise2 import format_alignment
from Bio.SubsMat.MatrixInfo import blosum62


torch.manual_seed(1)
warnings.simplefilter('ignore')
