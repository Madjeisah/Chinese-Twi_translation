# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 20:11:21 2019

@author: Adjeisah Michael
"""

# Import the libraries
#import numpy as np 
import re
from pickle import dump
import nltk
import glob
import os
from datetime import datetime
from sklearn.model_selection import train_test_split

nltk.download('stopwords')


# /////////////////////////////////////////////////////
# Importing dataset 1
# /////////////////////////////////////////////////////
file_list = glob.glob(os.path.join(os.getcwd(), "chi/all/*.txt"))
#file_list = glob.glob(os.path.join(os.getcwd(), "chi/oLDtEST/19_PSA.txt"))
chi_version = []
for file_path in file_list:
    with open(file_path, encoding='utf-8') as f_input:
        chi_version.append(f_input.read())
                
# split a loaded document into sentences 
# you can either use .strip().split('\n')
# or just .split('\n')
for line_e in chi_version:
    chi_sentence = line_e.split('\n')
    
# Get rid of empty lines   
chi_sentences = []
for line in chi_sentence:
    # Strip whitespace, should leave nothing if empty line was just "\n"
    if not line.strip():
        continue
    # We got something, save it
    else:
        chi_sentences.append(line)
        

# /////////////////////////////////////////////////////
# Importing dataset 2
# /////////////////////////////////////////////////////
file_list = glob.glob(os.path.join(os.getcwd(), "twi/all/*.txt"))
#file_list = glob.glob(os.path.join(os.getcwd(), "twi/oLDtEST/19_PSA.txt"))
twi_version = []
for file_path in file_list:
    with open(file_path, encoding='utf-8') as f_input:
        twi_version.append(f_input.read())
        
for line_t in twi_version:
    twi_sentence = line_t.strip().split('\n')

# Get rid of empty lines   
twi_sentences = []
for line in twi_sentence:
    # Strip whitespace, should leave nothing if empty line was just "\n"
    if not line.strip():
        continue
    # We got something, save it
    else:
        twi_sentences.append(line)
    
# Cleaning of the texts
def clean_txt(text):
    text = text.lower()
    text = re.sub(r'!',' ',text)
    text = re.sub(r',',' ',text)
    text = re.sub(r'﻿“',' ',text)
    text = re.sub(r'”',' ',text)
    text = re.sub(r'“',' ',text)
    text = re.sub(r'’',' ',text)
    text = re.sub(r'‘',' ',text)
    text = re.sub(r"[-()\"#/@;:<>{}+=~`、○|ʼ.?^0-9—]", " ", text)
    text = re.sub(r'—',' ',text)
    
    #this help to take care of multiple space
    text = " ".join(text.split())
    return text

# Clean the chinese version and save to txt
clean_chi_sent = []
for clean_ch in chi_sentences:
    clean_chi_sent.append(clean_txt(clean_ch))
    
with open ('dataset/all_chi.txt','w', encoding='utf-8') as proc_seqf:
    for i in clean_chi_sent:
        proc_seqf.write((i) + "\n") 
    
    
# Clean the twi version and save to txt
clean_twi_sent = []
for clean_tw in twi_sentences:
    clean_twi_sent.append(clean_txt(clean_tw))
    
with open ('dataset/all_twi.txt','w', encoding='utf-8') as proc_seqf:
    for i in clean_twi_sent:
        proc_seqf.write((i) + '\n') 


########### Process both text file into tab separated columns  ############
# Work on it
#with open(datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")+".txt", 'w', encoding='utf-8')as proc_seqf:
with open ('dataset/all_twi-chi.txt','w', encoding='utf-8') as proc_seqf:
    for i, j in zip(clean_twi_sent, clean_chi_sent):
        proc_seqf.write("{}\t{}".format(i, j) + "\n")   