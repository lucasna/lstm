
# coding: utf-8

# In[281]:


model_name = ""
train_size = 10000
hidden_units_nb = 100
layer_nb = 1
epoch_nb = 50
optimizer = "Adam"
max_words = 3000
batch=32

print("train_size = ", train_size)
print("hidden_units_nb = ", hidden_units_nb)
print("epoch_nb = ",epoch_nb)
print("optimizer = ",optimizer)
print("max_words = ",max_words)


# In[282]:


# keras module for building LSTM 
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.models import Sequential, load_model
import keras.utils as ku 

# set seeds for reproducability
from tensorflow import set_random_seed
from numpy.random import seed
set_random_seed(2)
seed(1)

import pandas as pd
import numpy as np
import string, os 

import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)


# In[283]:


curr_dir = 'data/'
f = open(curr_dir+'titles1M.txt','r')
content_all = f.readlines()
content = content_all[:train_size]
test_content = content_all[-100:]
# remove whitespace characters like `\n` at the end of each line
content = [x.strip() for x in content] 
content[:10]


# In[284]:


import re
def clean_text(txt):
    txt = re.sub(r'\([^)]*\)', '', txt)
    txt = txt.replace("Recipe", "")
    txt = txt.replace("&", "And")
    punct = string.punctuation
    punct = punct.replace("'", "")
    txt = "".join(v for v in txt if v not in punct)
    txt = txt.encode("utf8").decode("ascii",'ignore')
    return txt

corpus = [clean_text(x) for x in content]
corpus[:10]


# In[285]:


tokenizer = Tokenizer(num_words=max_words)

def get_sequence_of_tokens(corpus):
    ## tokenization
    tokenizer.fit_on_texts(corpus)
    total_words = min(max_words, len(tokenizer.word_index) + 1)
    print("total words = ",len(tokenizer.word_index) + 1)
    ## convert data to sequence of tokens 
    input_sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)
    return input_sequences, total_words

inp_sequences, total_words = get_sequence_of_tokens(corpus)

inp_sequences[:10]


# In[286]:


def generate_padded_sequences(input_sequences):
    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
    predictors, label = input_sequences[:,:-1],input_sequences[:,-1]
    label = ku.to_categorical(label, num_classes=total_words)
    return predictors, label, max_sequence_len

predictors, label, max_sequence_len = generate_padded_sequences(inp_sequences)


# In[287]:


def create_model(max_sequence_len, total_words):
    input_len = max_sequence_len - 1
    model = Sequential()
    
    # Add Input Embedding Layer
    model.add(Embedding(total_words, 10, input_length=input_len))
    
    # Add Hidden Layer 1 - LSTM Layer
    for i in range(1,layer_nb):
        model.add(LSTM(hidden_units_nb, return_sequences=True))
    model.add(LSTM(hidden_units_nb, return_sequences=False))
    
    # Add Output Layer
    model.add(Dense(total_words, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    
    return model

model = create_model(max_sequence_len, total_words)
model.summary()


# In[288]:


model.fit(predictors, label, epochs=epoch_nb, verbose=2, batch_size=batch)


# In[289]:


prepositions = ["with", "a", "and"]

def generate_text(seed_text, next_words, model, max_sequence_len):
    for i in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = model.predict_classes(token_list, verbose=0)
        
        output_word = ""
        for word,index in tokenizer.word_index.items():
            if index == predicted:
                if (i == next_words-1) & (word in prepositions):
                    continue
                output_word = word
                break
        seed_text += " "+output_word
    return seed_text.title()


# In[290]:


print (generate_text("Red", 4, model, max_sequence_len))
print (generate_text("Pumpkin", 4, model, max_sequence_len))
print (generate_text("Chocolate", 4, model, max_sequence_len))
print (generate_text("Spicy", 3, model, max_sequence_len))
print (generate_text("Pork", 4, model, max_sequence_len))
print (generate_text("Strawberry", 2, model, max_sequence_len))
print (generate_text("Potato", 2, model, max_sequence_len))
print (generate_text("Easy", 5, model, max_sequence_len))
print (generate_text("Roasted", 3, model, max_sequence_len))
print (generate_text("English", 5, model, max_sequence_len))
print (generate_text("Spaghetti", 6, model, max_sequence_len))


# In[291]:


#test loss

def test_get_sequence_of_tokens(corpus):
    ## tokenization
    tokenizer.fit_on_texts(corpus)
    total_words = min(max_words, len(tokenizer.word_index) + 1)
    print("total words = ",len(tokenizer.word_index) + 1)
    ## convert data to sequence of tokens 
    input_sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, min(max_sequence_len,len(token_list))):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)
    return input_sequences, total_words
def test_generate_padded_sequences(input_sequences):
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
    predictors, label = input_sequences[:,:-1],input_sequences[:,-1]
    label = ku.to_categorical(label, num_classes=total_words)
    return predictors, label, max_sequence_len

test_corpus = [clean_text(x) for x in test_content]
tokenizer = Tokenizer(num_words=max_words)
test_inp_sequences, test_total_words = test_get_sequence_of_tokens(test_corpus)
test_predictors, test_label, test_max_sequence_len = test_generate_padded_sequences(test_inp_sequences)
print("train_size = ", train_size)
print("hidden_units_nb = ", hidden_units_nb)
print("epoch_nb = ",epoch_nb)
print("optimizer = ",optimizer)
print("max_words = ",max_words)
print("test loss =", model.evaluate(test_predictors, test_label))


# In[189]:


if model_name == "":
    model_name = "tr"+str(train_size)+"ep"+str(epoch_nb)+str(optimizer)+"wd"+str(max_words)
    
model.save("models/"+model_name+".h5")
print("saved model to disk.")


# In[ ]:



#loaded_model = load_model('models/model4.h5')
#print("Loaded model from disk")

