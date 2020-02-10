#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 16:03:31 2020

@author: msr
"""

import pandas as pd
from collections import Counter
from nltk.tokenize import word_tokenize


filename = "recipes_82k.csv"

class DataGetter():
    
    """ This is a class that loads and transforms data so it can be 
    immediately usable with just the appropriate method calls.
    
    Also has attributes that contain extracted data.
    """
    
    
    def __init__(self, filename):
        
        self.df = pd.read_csv(filename)
        
        self.cooking_method = self.df['cooking_method'].to_list()
        
        self.cuisine = self.df['cuisine'].to_list()
        self.ingredients = self.df['ingredients'].to_list()
        
        self.prep_time = [[sublist] for sublist in self.df['prep_time'].to_list()]
        
        self.recipe_name = [[sublist] for sublist in self.df['recipe_name'].to_list()]
        
        self.serves = [[sublist] for sublist in self.df['serves'].to_list()]
        self.tags = [[sublist] for sublist in self.df['tags'].to_list()]
        
        self.tags_count = Counter([tag for taglist in self.tags for tag in taglist])
            
    
    
    def get_sentences(self):
        
        """ Returns a list of sentences in plain text"""
        
        self.sentences = self.df.text.tolist()
        return self.sentences



    def clean_and_tokenise(self, colname):   
        
        """ Returns a list of tokenised and cleaned sentences"""
        
        coldata = df['cooking_method'].to_list()
        
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
            
    
    def get_unique_entities(self):
        """Returns list of unique entities"""
        self.unique_ents = list(set([dict_['entity'] for sentence in self.df.entities for dict_ in sentence]))
        return self.unique_ents

        
    def get_unique_intents(self):
        """Returns list of unique intents"""
        self.unique_intents = list(set(self.df.intent))
        return self.unique_intents
        
    
    def get_entities(self):
        
        """Returns matching entity for every word"""
        
        self.clean_and_tokenise()
        
        self.entitites_start_stop = df.entities.apply(lambda x: [(w['start'], w['stop']) for w in x])
    
        self.entity_sequence = list(map(lambda x: ['O' for word in x], self.sentences_tokenised))
        
        for i, sent in enumerate(self.sentences_tokenised):
            for ent in self.df.iloc[i].entities:
                start = ent['start']
                stop = ent['stop']
                
                entity = ent['entity']
                
                self.entity_sequence[i][start:stop+1] = [entity for i in 
                                                         range(len(self.entity_sequence[i][start:stop+1]))]


        return self.entity_sequence
