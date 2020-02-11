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
nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

import string
punctuations = '!"#$%&\'()*+,-/:;<=>?@[\\]^_`{|}~'
table = str.maketrans('', '', punctuations)

import difflib


from tqdm import tqdm

df = pd.read_csv("recipes_82k.csv")

df.drop(["category", "image"], inplace=True, axis=1)

#Handling missing data
df['cuisine'].fillna('UNK_Cuisine', inplace=True)
df['prep_time'].fillna('UNK_Time', inplace=True)
df['serves'].fillna('UNK_Serves', inplace=True)
#df['tags'].fillna("Unk_Tag", inplace=True)
df.dropna(inplace = True)
df = df.where(pd.notnull(df), None)


df = df[df.duplicated() == False]


cooking_method = df['cooking_method']

ingredients = list(df['ingredients'])

prep_time = list(df['prep_time'])

recipe_name = list(df['recipe_name'])

serves = list(df['serves'])

tags = list(df['tags'])
tags = [str(tag).split(",") for tag in tags]

tags_flat = [tag for tag_list in tags for tag in tag_list]

tags_unique = list(set(tags_flat))


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

bb = []

for elem in tqdm(ingredients):
    elem = elem.translate(table)
    # elem = lemmatizer.lemmatize(elem)
    # elem = word_tokenize(elem)
    # elem = [word.translate(puncts) for word in elem]
    elem = [word.lower() for word in elem if word not in stop_words]
    
    elem = ' '.join(elem)

    # for i, word in enumerate(elem):
    #     if elem[i] == '' and elem[i+1] == '':
    #         del elem[i+1]
    
    bb.append(elem)

bb = []

for ing_list in tqdm(ingredients):
    simplified_ing_list = []
    # ing = ing.split(",")
    # ing_list = ing_list.translate(table)
    ing_list = ing_list.split(",")
    
    for i, ing in enumerate(ing_list):
        overlap = difflib.get_close_matches(ing, simplified_ingredients, 1)
        if overlap:
            simplified_ing_list.append(overlap)
        
        # overlap_indices = np.flatnonzero(np.core.defchararray.find(ing, simplified_ingredients) != -1)
        # if overlap_indices:
        #     len_matches = []
        #     for oi in overlap_indices:
                
            
        #     index_in_simplified = np.where(simplified_ingredients == "fennel")
        #     ing_list[i] = simplified_ingredients[index_in_simplified]
        # else:
        #     del ing_list[i]
    
    # ing_list = [ing for ing in simplified_ingredients if org_ing in simplified_ingredients]
    
    aa.append(simplified_ing_list)



aa = []

for elem in tqdm(cooking_method):
    aa.append(nlp(elem))




    
# for i, tag in enumerate(tags):
#     tag = str(tag).split(",")
#     tags_flat.append(tag)

    
    