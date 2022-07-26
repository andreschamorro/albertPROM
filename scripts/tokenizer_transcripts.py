#!/usr/bin/env python
# coding: utf-8

# In[1]:


from transformers import GPT2Config, TFGPT2Model
from tokenizers import ByteLevelBPETokenizer

# Initialize a GPT2 Model
gpt2_config = GPT2Config()
model = TFGPT2Model(gpt2_config)
# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer()


# In[2]:


# importing module
import sys
# appending a path
sys.path.extend(['.', '..'])


# In[3]:


import datasets

trans_data = datasets.load_dataset('../loaders/dataset_script.py', data_dir='data')
trans_data = trans_data.shuffle(seed=42)
trans_data = trans_data.map(lambda seq: {"feature": seq["feature"].upper()}, num_proc=4)


# In[4]:


import random

k_low = 12
k_high = 12

def kmernizer(seq):
    return " ".join([seq[i: i + random.randint(k_low, k_high + 1)] for i in range(len(seq) - k_high + 1)])

def batch_kmer(batch):
    return [kmernizer(seq) for seq in batch]


# In[7]:


trans_data = trans_data.map(lambda seq: {"feature": kmernizer(seq["feature"])}, num_proc=4)


# In[8]:


batch_size = 1000

def get_training_corpus():
    for i in range(0, len(trans_data["train"]), batch_size):
        yield trans_data["train"][i : i + batch_size]["feature"]


# In[ ]:


tokenizer.train_from_iterator(get_training_corpus(), vocab_size=6781480)


# In[ ]:


tokenizer.save_pretrained("transcripts_gpt2_tokenizer")

