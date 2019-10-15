# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 13:49:08 2019

@author: Adjeisah Michael
"""

# Import libraries
#from numpy import array
from numpy.random import shuffle
from sklearn.model_selection import train_test_split
import os
import sentencepiece as spm
import logging
from hparams import Hparams
import time

logging.basicConfig(level=logging.INFO)

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, mode='rt', encoding='utf-8')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# load dataset
filename = 'dataset/all_twi-chi.txt'
processed_seq = load_doc(filename)


# split a loaded document into sentences
def to_pairs(processed_seq):
	lines = processed_seq.strip().split('\n')
	pairs = [line.split('\t') for line in  lines]
	return pairs

# split into english-german pairs
pairs = to_pairs(processed_seq)
shuffle(pairs)


# Spliting the data into training and testing: test_size = 069
# to contains approximately 29003 training examples, 
# 800 validation examples, and 13000 test examples.
train, test = train_test_split(pairs, test_size=0.0675, random_state=101)


# Spliting the testing set into testing and validation set but before call 
# call shuffle on the test set
# Wait for 5 seconds test_size = 0.5235 gives you exactely 1000 val
time.sleep(1)
shuffle(test)
val, test = train_test_split(test, test_size=0.619, random_state=101)

# spot checking
for i in range(5):
	print('[%s] => [%s]' % (train[i][0], train[i][1]))
    
# spot check and save to training 
train_twi = []
train_chi = []
for i in range(len(train)):
    #print(train[i][0])
    train_twi.append(train[i][0])
    train_chi.append(train[i][1])
    
# spot check and save to testing
test_twi = []
test_chi = []
for i in range(len(test)):
    test_twi.append(test[i][0])
    test_chi.append(test[i][1])

# spot check and save to avaluation
val_twi = []    
val_chi = []
for i in range(len(val)):
    val_twi.append(val[i][0]) 
    val_chi.append(val[i][1])  

logging.info("# Preprocessing")
assert len(train_twi)==len(train_chi), "Check if train source and target files match."  
assert len(test_twi) == len(test_chi), "Check if test source and target files match." 
assert len(val_twi) == len(val_chi), "Check if eval source and target files match." 

logging.info("Let's see how preprocessed data look like")
logging.info("train_twi:", train_twi[0])
logging.info("train_chi:", train_chi[0])
logging.info("test_twi:", test_twi[0])
logging.info("test_chi:", test_chi[0])
logging.info("val_twi:", val_twi[0])
logging.info("val_chi:", val_chi[0])

def _write(sents, fname):
    with open(fname, 'w', encoding='utf-8') as fout:
        fout.write("\n".join(sents))
#sep = ["-----------------------------@@@@@<==>@@@@-----------------------------"]
_write(train_twi, 'dataset/split/train.twi')  
_write(train_chi, 'dataset/split/train.chi') 
#_write(train_twi+sep+train_chi, 'dataset/split/train') 
_write(train_twi+train_chi, 'dataset/split/train') 
_write(test_twi, 'dataset/split/test.twi') 
_write(test_chi, 'dataset/split/test.chi') 
_write(val_twi, 'dataset/split/val.twi') 
_write(val_chi, 'dataset/split/val.chi')



def prepro(hp):
    logging.info("# Train a joint BPE model with sentencepiece")
    os.makedirs("dataset/segmented", exist_ok=True)
    _train = '--input=dataset/split/train --pad_id=0 --unk_id=1 \
             --bos_id=2 --eos_id=3\
             --model_prefix=dataset/segmented/bpe --vocab_size={} \
             --model_type=bpe'.format(hp.vocab_size)
    spm.SentencePieceTrainer.Train(_train)
    
    logging.info("# Load trained bpe model")
    sp = spm.SentencePieceProcessor()
    sp.Load("dataset/segmented/bpe.model")
    
    logging.info("# Segment")
    def _segment_and_write(sents, fname):
        with open(fname, "w", encoding='utf-8') as fout:
            for sent in sents:
                pieces = sp.EncodeAsPieces(sent)
                fout.write(" ".join(pieces) + "\n")
                
    
    _segment_and_write(train_twi, "dataset/segmented/train.twi.bpe")
    _segment_and_write(train_chi, "dataset/segmented/train.chi.bpe")
    _segment_and_write(val_twi, "dataset/segmented/val.twi.bpe")
    _segment_and_write(val_chi, "dataset/segmented/val.chi.bpe")
    _segment_and_write(test_twi, "dataset/segmented/test.twi.bpe")
    
    
    logging.info("Let's see how segmented data look like")
    print("train1:", open("dataset/segmented/train.twi.bpe",'r', encoding='utf-8').readline())
    print("train2:", open("dataset/segmented/train.chi.bpe", 'r', encoding='utf-8').readline())
    print("eval1:", open("dataset/segmented/val.twi.bpe", 'r', encoding='utf-8').readline())
    print("eval2:", open("dataset/segmented/val.chi.bpe", 'r', encoding='utf-8').readline())
    print("test1:", open("dataset/segmented/test.twi.bpe", 'r', encoding='utf-8').readline())
    
if __name__ == '__main__':
    hparams = Hparams()
    parser = hparams.parser
    hp = parser.parse_args()
    prepro(hp)
    logging.info("Done")
    