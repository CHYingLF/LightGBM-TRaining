import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import random
import numpy as np
from models.models import lgbm_model_search

# Data
train = pd.read_csv('../data/train.csv')
labels = pd.read_csv('../data/train_labels.csv')

not_features = ['sequence', 'subject', 'step', 'state']

df = pd.read_csv('../data/train_df.csv')
df = df.merge(labels, on = 'sequence', how = 'left')

train_df, val_df = train_test_split(df, train_size= 0.95, random_state = 44, stratify= df['state'])

train_df[['sequence']].to_csv('../data/train_id.csv', index = 0)
val_df[['sequence']].to_csv('../data/val_id.csv', index = 0)

features = [f for f in df.columns if not f in not_features]

X_train, Y_train, X_val, Y_val = train_df[features], train_df['state'], val_df[features], val_df['state']

# Model
model, pred = lgbm_model_search(X_train, Y_train, X_val, Y_val)
confusion = confusion_matrix(pred, Y_val.values)
print(confusion)