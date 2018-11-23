
# coding: utf-8

# In[278]:


model_name = ""
train_size = 200
hidden_units_nb = 20
epoch_nb = 5
optimizer = "Adam"
max_words = 5000

print("train_size = ", train_size)
print("hidden_units_nb = ", hidden_units_nb)
print("epoch_nb = ",epoch_nb)
print("optimizer = ",optimizer)
print("max_words = ",max_words)


# In[279]:


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


# In[280]:


curr_dir = 'data/'
f = open(curr_dir+'tokenized_instructions_val.txt','r')
content_all = f.readlines()
content = content_all[:train_size]
test_content = content_all[-100:]
content[:1]
# remove whitespace characters like `\n` at the end of each line
content = [x.strip() for x in content] 
content[:1]


# In[281]:


import re
def clean_text(txt):
    txt = txt.encode("utf8").decode("ascii",'ignore')
    return txt

corpus = [clean_text(x) for x in content]
corpus[:1]


# In[282]:


tokenizer = Tokenizer(num_words=max_words, filters='"#$%()*+-<=>?@[\\]^_`{|}~\t\n') #do not filter the dot, virgule, &, colunms, !, /

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


# In[284]:


def generate_padded_sequences(input_sequences):
    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
    predictors, label = input_sequences[:,:-1],input_sequences[:,-1]
    label = ku.to_categorical(label, num_classes=total_words)
    return predictors, label, max_sequence_len

predictors, label, max_sequence_len = generate_padded_sequences(inp_sequences)


# In[285]:


def create_model(max_sequence_len, total_words):
    input_len = max_sequence_len - 1
    model = Sequential()
    
    # Add Input Embedding Layer
    model.add(Embedding(total_words, 10, input_length=input_len))
    
    # Add Hidden Layer 1 - LSTM Layer
    model.add(LSTM(hidden_units_nb, return_sequences=True))
    model.add(LSTM(hidden_units_nb))
    
    # Add Output Layer
    model.add(Dense(total_words, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    
    return model

model = create_model(max_sequence_len, total_words)
model.summary()


# Lets train our model now

# In[286]:


model.fit(predictors, label, epochs=epoch_nb, verbose=2)


# In[287]:


def generate_text(seed_text, next_words, model, max_sequence_len):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = model.predict_classes(token_list, verbose=0)
        
        output_word = ""
        for word,index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " "+output_word
    return seed_text.title()


# In[288]:


print (generate_text("Pour", 20, model, max_sequence_len))
print (generate_text("Saute", 20, model, max_sequence_len))
print (generate_text("Blend", 20, model, max_sequence_len))
print (generate_text("Prepare", 20, model, max_sequence_len))
print (generate_text("Preheat", 20, model, max_sequence_len))
print (generate_text("Preheat", 20, model, max_sequence_len))
print (generate_text("1", 20, model, max_sequence_len))
print (generate_text("Stir", 20, model, max_sequence_len))
print (generate_text("In", 20, model, max_sequence_len))
print (generate_text("Preheat", 20, model, max_sequence_len))
print (generate_text("Cream", 20, model, max_sequence_len))


# In[289]:


if model_name == "":
    model_name = "I"+"tr"+str(train_size)+"ep"+str(epoch_nb)+str(optimizer)+"wd"+str(max_words)
    
model.save("models/"+model_name+".h5")
print("saved model to disk.")


# In[290]:


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

test_corpus = [clean_text(x) for x in test_content]
tokenizer = Tokenizer(num_words=max_words)
test_inp_sequences, test_total_words = test_get_sequence_of_tokens(test_corpus)
test_predictors, test_label, test_max_sequence_len = generate_padded_sequences(test_inp_sequences)
print(test_predictors.shape)
model.evaluate(test_predictors, test_label)


# In[ ]:



#loaded_model = load_model('models/model4.h5')
#print("Loaded model from disk")

