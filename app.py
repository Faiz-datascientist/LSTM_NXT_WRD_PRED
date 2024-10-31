import pickle 
import numpy as np
import pandas as pd
import tensorflow as tf
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


#Loading the lstm model

model=load_model('lstm_model.h5')


#Load Tokenizer

with open('tokenizer.pickle','rb') as tok:
    tokenizer=pickle.load(tok)



def next_word_predict(model,tokenizer,text,max_seq_len):
    #tokenize
    list_token=tokenizer.texts_to_sequences([text])[0]
    #truncate
    if len(list_token) >=max_seq_len:
        token_list = token_list[-(max_seq_len-1):] # [Truncate] - we are taking last tokens so that we can predict next words
    #padding
    token_list=np.array(pad_sequences([list_token],maxlen=max_seq_len-1,padding='pre'))

    #predict
    predicted=model.predict(token_list,verbose=0)
    predicted_word_index=np.argmax(predicted,axis=1) #max value of index
    for word,index in tokenizer.word_index.items():
        if index==predicted_word_index:
            return word
    
    return None






#Streamlit webApp

st.title("Next Word Prediction")
input_text=st.text_input("Enter Here!")
if st.button('Click'):
    max_sequence_length=model.input_shape[1]+1  #we have total 14 words but during model create we substract -1 now adding +1 for next space 
    next_word=next_word_predict(model,tokenizer,input_text,max_sequence_length)
    st.write(f"next word prediction is :{next_word}")
