#!/usr/bin/env python
# coding: utf-8

# # TODOs
# 1. Community prediction is systematically biased and underconfident. You need to quantify by how much and exploit it!

# In[1]:


import ergo
import datetime
import seaborn
import numpy as np
import pandas as pd

from KEYS import USERNAME, PASSWORD

import warnings
warnings.filterwarnings("ignore")

# ## Parameters

# In[2]:


CATS = [
    # "series--maximum-likelihood",
    "series--hill-climbing",
    "series--deep-learning"
]

# 20 qs/page, so this assumes <40 questions per round, in line with the "approximately 30" stated in the rules
# change accordingly if it fails
N_PAGES = 2

# kinda slow but it's worth it
N_SAMPLES = int(10e3)


# ## Main functions

# In[3]:


is_continuous = lambda q: q.data['possibilities']['type'] == 'continuous'
is_binary = lambda q: q.data['possibilities']['type'] == 'binary'
is_open = lambda q:    datetime.datetime.strptime(q.data['publish_time'], '%Y-%m-%dT%H:%M:%SZ') <    datetime.datetime.utcnow() <    datetime.datetime.strptime(q.data['close_time'], '%Y-%m-%dT%H:%M:%SZ')

def default_to_community_cont(question, n_samples):
    """Defaults to community distribution for continuous questions"""
    samples = np.array([question.sample_community() for _ in range(n_samples)])
    question.submit_from_samples(samples)

def default_to_community_bin(question):
    """Defaults to community prediction for binary questions"""
    p = question.get_community_prediction()
    question.submit(p)

def default_to_community(question, n_samples=None):
    if is_open(question):
        if is_binary(question):
            default_to_community_bin(question)
        elif is_continuous:
            if n_samples is None:
                raise ValueError("n_samples can't be None for continuous questions")
            default_to_community_cont(question, n_samples)
        else:
            print(f"Question {question.id} is of type {q.data['possibilities'].get('type')} and cannot be predicted")
    else:
        print(f"Question {question.id} is not open!")


# ## Login

# In[4]:


metaculus = ergo.Metaculus()
metaculus.login_via_username_and_password(
    username=USERNAME,
    password=PASSWORD
)


# ## Load questions

# In[5]:


qs = []
for cat in CATS:
    qs.extend(metaculus.get_questions(cat=cat, pages=N_PAGES))


# ## Submit

# In[6]:


for q in qs:
    print(f"{q.id}: {q.name}")
    try:
        default_to_community(q, n_samples=N_SAMPLES)
    except Exception as e:
        print(e)
        continue

