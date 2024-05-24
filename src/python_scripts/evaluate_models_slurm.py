from sentence_transformers import SentenceTransformer, InputExample, losses, models
from sentence_transformers.datasets import DenoisingAutoEncoderDataset
from torch.utils.data import DataLoader
import random
import pandas as pd
import torch
import numpy as np
import gc
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, log_loss

testPath = "/d/hpc/home/mm4195/data/raw/Annotation_sentence-level_utf8.txt"
df = pd.read_csv(testPath, sep="\t")
df = df.iloc[:, [3, 12]]

df = df.iloc[:38000, :]

X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, 0], df.iloc[:, 1], train_size=0.8, random_state=42)


modelss= ["tsdae-hate_speech_slo_gpl", "tsdae-hate_speech_slo"]

base_model_path = "/d/hpc/home/mm4195/models_final/"

d = {}

for model_name in modelss:
    
    # model testing

    bbert = models.Transformer(base_model_path + model_name)
    bpooling = models.Pooling(bbert.get_word_embedding_dimension(), 'cls')
    model = SentenceTransformer(modules=[bbert, bpooling])

    print("Model: ", model_name)
    embedded_train = model.encode(X_train.to_numpy())
    embedded_test = model.encode(X_test.to_numpy())

    lr = LogisticRegression(max_iter=1000)
    lr.fit(embedded_train, y_train.to_numpy())

    y_pred = lr.predict(embedded_test)
    y_pred_probabs = lr.predict_proba(embedded_test)

    f1 = f1_score(y_test, y_pred, average='weighted')
    logloss = log_loss(y_test, y_pred_probabs)
    print("F1 score: ", f1)
    print("Log loss: ", logloss)
    d[model_name] = {"F1" :f1,"Log-Loss": logloss}

    torch.cuda.empty_cache() 


scores = pd.DataFrame(d).T
print(scores)
scores.to_csv("scores.csv")