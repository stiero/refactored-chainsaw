#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 14:33:22 2020

@author: msr
"""

import pandas as pd
import seaborn as sns
import string
from collections import Counter
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')

df = pd.read_csv("recipes_82k.csv")

cooking_method = list(df['cooking_method'])

ingredients = list(df['ingredients'])

prep_time = list(df['prep_time'])

recipe_name = list(df['recipe_name'])

serves = list(df['serves'])

tags = list(df['tags'])


Counter(tags)



def clean_and_tokenise(list_):   
    results = []
    for sublist in list_:
        sublist = word_tokenize(sublist)
        sublist[0].translate(None, string.punctuation)
        results.append(sublist)
    return results



aa = instructions[0]

bb = ingredients[0]


import spacy

nlp = spacy.load("en_core_web_lg")

columns = list(df.columns)


for elem in cooking_method:
    doc = nlp(elem)
    
    
def intersection(lst1, lst2): 
    lst3 = [value for value in lst1 if value in lst2] 
    return lst3 