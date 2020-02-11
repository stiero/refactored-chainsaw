#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 19:32:33 2020

@author: tauro
"""

import numpy as np

with np.load('simplified-recipes-1M.npz') as data:
    simplified_ingredients = data['ingredients']


ft_cooking = aa
ft_ingredients = bb
    
import pandas as pd

ft_df = pd.DataFrame(list(zip(ft_cooking, ft_ingredients)))


ft_df.to_csv("ft_df.csv", header=False, index=False)

cc = aa + bb


import fasttext
from gensim.models.fasttext import FastText


model = FastText(cc, size=100, window=10, iter=10)

model.wv.most_similar("olive")


from gensim.test.utils import get_tmpfile

fname = get_tmpfile("fasttext.model")

model.save("fasttext.model")

model1 = FastText.load('fasttext.model')
