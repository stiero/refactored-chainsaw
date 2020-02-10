#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 14:33:22 2020

@author: msr
"""

import pandas as pd
import seaborn as sns
from collections import Counter

df = pd.read_csv("recipes_82k.csv")

cooking_method = list(df['cooking_method'])

ingredients = list(df['ingredients'])

prep_time = list(df['prep_time'])

recipe_name = list(df['recipe_name'])

serves = list(df['serves'])

tags = list(df['tags'])


Counter(tags)



def clean_and_tokenise(self):   
        
        """ Returns a list of tokenised and cleaned sentences"""
        
        self.get_sentences()
        
        """Removes '!', and '.' punctuations from sentence strings
        """
        for i, sent in enumerate(self.sentences):
            self.sentences[i] = re.sub('[.!]+', "", sent)            
            
        """ Splits string based on (- ' : , and ?). Deduced this list by 
            looking at the start-stop indices at various examples
        """
        self.sentences_tokenised = list(map(lambda x: re.split(" |(-)|'|(:)|(,)|(\?)", x), self.sentences))
        
        for i, sent in enumerate(self.sentences_tokenised):
            self.sentences_tokenised[i] = list(filter(None, sent))
            
        return self.sentences_tokenised




aa = instructions[0]

bb = ingredients[0]


import spacy

nlp = spacy.load("en_core_web_lg")

columns = list(df.columns)
