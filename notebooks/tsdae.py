from sentence_transformers import SentenceTransformer, InputExample, losses, models
from sentence_transformers.datasets import DenoisingAutoEncoderDataset
from torch.utils.data import DataLoader
import random
import pandas as pd
import torch
import numpy as np
import gc
import nltk 
nltk.download('punkt')



print("Total GPU Memory:", torch.cuda.get_device_properties(0).total_memory / 1e9, "GB")
print("Available GPU Memory:", torch.cuda.memory_reserved(0) / 1e9, "GB")
print("Allocated GPU Memory:", torch.cuda.memory_allocated(0) / 1e9, "GB")
print("Free (cached) GPU Memory:", torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0) / 1e9, "GB")

df = pd.read_csv('~/data/Annotation_sentence-level_utf8.txt', sep='\t')
df = df.dropna()

paragrah = df.iloc[:, [3, 12]].to_numpy()

basePath = "/d/hpc/home/ak3883/models/"
names = ["bert-base-multilingual-uncased", "all-MiniLM-L12-v2", "hate_speech_slo"]

df = pd.read_csv('~/data/training_phrases.csv', header=None)
sentences = df.iloc[:, 0].to_numpy()

train_data = DenoisingAutoEncoderDataset(sentences)
train_dataloader = DataLoader(train_data, batch_size=16, shuffle=True, drop_last=True)

for modelName in names:
    torch.cuda.empty_cache() 
    print("Training model: ", modelName)
    modelPath = basePath + modelName

    bert = models.Transformer(modelPath)
    pooling = models.Pooling(bert.get_word_embedding_dimension(), 'cls')
    model = SentenceTransformer(modules=[bert, pooling])

    print(model)

    print("losses")

    train_loss = losses.DenoisingAutoEncoderLoss(model,decoder_name_or_path=modelPath ,tie_encoder_decoder=False)

    num_epochs = 1

    print('Training the model')
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=num_epochs,
        weight_decay=0,
        scheduler='constantlr',
        optimizer_params={'lr': 3e-5},
        show_progress_bar=True
    )

    model.save(basePath+"tsdae-" + modelName)
