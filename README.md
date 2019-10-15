# Chinese-Twi_translation
 NMT experiments on Chinese-Twi parallel Bible corpus with the state-of-the-art Transformer model, a self-attention encoder-decoder model.
 A sentence level alignment technique has been employed for creating a bilingual Chinese-Twi Corpus dataset. The first attempt on 
 Chinese-Twi corpora with about 31103 parallel-aligned sentences.

# Usage
Please refer to train.py and test.py

### train.py
This task is same as in A TensorFlow Implementation of the Transformer: Attention Is All You Need (https://github.com/Kyubyong/transformer).
We borrowed the codes in the repository, and then tweaked to fit the demands of this language pair

#### Results
 A BLEU score of 76.9% was achieved on the train dataset, and 61.5% on the test set. Results and findings of this work are considerable 
 and serve as a base line for the future of machine translation (MT) for this language pair.valid accuracy. 
 We observe that for a small amout of dataset using smaller model parameters such as layers=4 and d_model=256, is better 
 since the task is quite small.

### For your own data
Just preproess your source and target sequences in parallel format seperted by tab dilimeted. Refer to the file text_pro_to_seq.py.

## Upgrades
-Reconstruct some classes.
It is more easier to use the components in other models in (https://github.com/Kyubyong/transformer)

# Acknowledgement
Some model structures and some scripts are borrowed from https://github.com/Kyubyong/transformer
