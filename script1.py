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
nltk.download('stopwords')

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(["minutes", "hours", "seconds", "teaspoon", "spoon"])

import string
punctuations = '!"#$%&\'()*+,.-/:;<=>?@[\\]^_`{|}~'
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

#Random shuffle
df = df.sample(frac=1, random_state=1234)


cooking_method = df['cooking_method']

cuisine = list(df['cuisine'])

ingredients = list(df['ingredients'])

prep_time = list(df['prep_time'])

recipe_name = list(df['recipe_name'])

serves = list(df['serves'])

tags = list(df['tags'])
tags = [str(tag).split(",") for tag in tags]

tags_flat = [tag for tag_list in tags for tag in tag_list]

tags_unique = list(set(tags_flat))


Counter(tags_flat)



def clean_and_tokenise(list_):   
    results = []
    for sublist in list_:
        sublist = word_tokenize(sublist)
        sublist[0].translate(None, string.punctuation)
        results.append(sublist)
    return results



# aa = instructions[0]

# bb = ingredients[0]


import spacy

nlp = spacy.load("en_core_web_lg")

columns = list(df.columns)

tok_cooking_method = []

for elem in tqdm(cooking_method):
    elem = elem.translate(table)
    elem = lemmatizer.lemmatize(elem)

    elem = word_tokenize(elem)
    # elem = [word.translate(puncts) for word in elem]
    elem = [word.lower() for word in elem if word not in stop_words and word.isalpha() and not word.endswith("ly")]
    
    # elem = ' '.join(elem)

    # for i, word in enumerate(elem):
    #     if elem[i] == '' and elem[i+1] == '':
    #         del elem[i+1]
    
    tok_cooking_method.append(elem)



tok_ingredients = []

for elem in tqdm(ingredients):
    elem = elem.translate(table)
    elem = lemmatizer.lemmatize(elem)
    # elem = elem.split(",")
    elem = word_tokenize(elem)
    # elem = [word.translate(puncts) for word in elem]
    elem = [word.lower() for word in elem if word not in stop_words and word.isalpha() and not word.endswith("ly")]
    
    # elem = ' '.join(elem)

    # for i, word in enumerate(elem):
    #     if elem[i] == '' and elem[i+1] == '':
    #         del elem[i+1]
    
    tok_ingredients.append(elem)



tok_recipe_name = []

for elem in tqdm(recipe_name):
    elem = elem.translate(table)
    elem = lemmatizer.lemmatize(elem)
    elem = word_tokenize(elem)
    elem = [word.lower() for word in elem if word not in stop_words and word.isalpha()]
    
    tok_recipe_name.append(elem)




#Length of text
    
len_cooking = []
for subl in tok_cooking_method:
    len_cooking.append(len(subl))

max(len_cooking)

len_ingredients = []

for subl in tok_ingredients:
    len_ingredients.append(len(subl))
    
max(len_ingredients)


len_recipe_name = []

for subl in tok_recipe_name:
    len_recipe_name.append(len(subl))
    
max(len_recipe_name)


import seaborn as sns
sns.kdeplot(len_cooking)
sns.kdeplot(len_ingredients)



for i, subl in enumerate(tok_cooking_method):
    subl = subl[:200]
    tok_cooking_method[i] = subl
    
for i, subl in enumerate(tok_ingredients):
    subl = subl[:100]
    tok_ingredients[i] = subl
    
for i, subl in enumerate(tok_recipe_name):
    subl = subl[:10]
    tok_recipe_name[i] = subl


tok_concat = []

for i, j, k in zip(tok_cooking_method, tok_ingredients, tok_recipe_name):
    tok_concat.append(i + j + k)






aa = []

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

    
    
    
import keras.backend as K
import numpy as np
    
total_size = len(len_cooking)
    
train_indices = list(range(0,int(total_size*0.8)-1))
train_size = len(train_indices)

val_indices = list(range(max(train_indices), total_size))
val_size = len(val_indices)

max_text_len = 310
vector_size = 100

X_train = np.zeros((train_size, max_text_len, vector_size), dtype=K.floatx())

X_val = np.zeros((val_size, max_text_len, vector_size), dtype=K.floatx())


for index in range(0, total_size):
    for t, token in enumerate(tok_concat[index]):
        if t >= max_text_len:
            break
        
        
        if index < train_size:
            X_train[index, t, :] = vecs[token]
            
        else:
            X_val[index-train_size, t, :] = vecs[token]
            
            


ntags = len(tags_unique)

tags2id = {tag:i for i, tag in enumerate(tags_unique)}

#y_train = [[0]*len(tags_unique)]*total_size

y = np.zeros((total_size, ntags), dtype=np.int8)

for i, tag_list in enumerate(tags):
    for j, tag in enumerate(tag_list):
        if tag in tags2id.keys():
            k = tags2id[tag]
            y[i][k] = 1
        
        # for k in tags2id.values():
        #     if j == tags2id[tag]:
        #         y_train[i][k] = 1


# from keras.utils import to_categorical

# y_train = []

# for tag_list in tags:
#     y = [to_categorical(tags2id[i], num_classes=ntags) for i in tag_list]
#     y_train.append(y)
    
    
    
tot_tags = len(tags_flat)
tag_weights = dict(Counter(tags_flat))

tag_weights = {tags2id[k]:tot_tags/v for k,v in tag_weights.items()}



y_train = y[:train_size]
y_val = y[train_size-1:]

# cuisine_train = np.array(cuisine[:train_size]).reshape(-1,1)
# cuisine_val = np.array(cuisine[train_size-1:]).reshape(-1,1)


from keras.preprocessing.sequence import pad_sequences
import numpy as np

from keras.callbacks import EarlyStopping
from keras.models import Input, Model
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers import Embedding, Bidirectional, LSTM, \
    GlobalMaxPooling1D, GlobalAveragePooling1D, Concatenate, concatenate, \
        TimeDistributed, MaxPooling1D, add
from keras.layers.convolutional import Conv1D
from keras.optimizers import Adam

from keras.losses import sparse_categorical_crossentropy

import tensorflow as tf


batch_size = 32
nb_epochs = 100

# n_cuisines = len(set(cuisine))
# input_cuisine_vecs = Input(shape=(1,))
# cuisine_vecs = Embedding(input_dim=n_cuisines, output_dim=100)(input_cuisine_vecs)



input_ = Input(shape=(max_text_len,vector_size))


model = TimeDistributed(Dense(256, use_bias=False, activation='elu'))(input_)
# model = Conv1D(filters=32, kernel_size=8, strides=1, 
#                activation='relu', padding='valid')(input_)
# model = MaxPooling1D(pool_size=4)(model)
# model = concatenate([input_, cuisine_vecs], axis=1)
model = Bidirectional(LSTM(units=50, recurrent_dropout=0.1, return_state = True, return_sequences=True))(model)
model = Bidirectional(LSTM(units=50, recurrent_dropout=0.2))(model)
# avg_pool = GlobalAveragePooling1D()(model)
# max_pool = GlobalMaxPooling1D()(model)
# # model = Bidirectional(LSTM(units=50, recurrent_dropout=0.1, return_sequences=True))(model)
# conc = concatenate([avg_pool, max_pool])
# conc = add([model, input_])
# model = Flatten()(model)
model = Dense(64, activation="elu")(model)
model = Dropout(0.1)(model)
out = Dense(ntags, activation="sigmoid")(model)
model = Model(input_, out)


model.summary()

model.compile(loss='binary_crossentropy',
              optimizer=Adam(lr=0.0001, decay=1e-6),
              metrics=['accuracy'])

history = model.fit(X_train, y_train,
              batch_size=batch_size,
              shuffle=True,
              epochs=nb_epochs,
              validation_data=(X_val, y_val),
              callbacks=[EarlyStopping(min_delta=0.00025, patience=2,
                                       monitor='val_loss')],
              class_weight = tag_weights,
              verbose=1) 




import matplotlib.pyplot as plt
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)

plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

from keras.utils import plot_model
plot_model(model, to_file = 'model.png')

model.save_weights("model_new_vector.h5")

from keras.models import load_model
model = load_model("model_new_vector.h5")

from scipy import spatial

1 - spatial.distance.cosine(vecs['orange'], vecs['oranges'])



cutoff = 0.001

for pred in preds:
    for val in pred:
        if cutof

preds = model.predict(X_val, batch_size=32)

testindex = 9283

aa = preds.copy()


aa[aa >= 0.1] = 1
aa[aa < 0.1] = 0


print(np.logical_and(aa[testindex]==1, y_val[testindex]==1).sum())

print(len(np.argwhere(aa[testindex])), len(np.argwhere(y_val[testindex])))
