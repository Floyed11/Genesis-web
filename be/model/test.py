import torch
from torch import nn
import pandas as pd
import pymongo
import pickle
import os

with open('model/columns.pkl', 'rb') as f:
                print('load columns success')
                load_columns = pickle.load(f)
df = load_columns
df = pd.DataFrame(df)
print(df)
df = df[df.index.duplicated()]
print(df)