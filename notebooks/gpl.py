import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM, Trainer, TrainingArguments
import pandas as pd
import os

# Set seed for reproducibility
seed = 42
torch.manual_seed(seed)

# Load training phrases from CSV file
df = pd.read_csv('~/data/training_phrases.csv', header=None)
sentences = df.iloc[:, 0].tolist()

# Function to tokenize data using a pre-trained tokenizer
def tokenize_data(model_name, sentences):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized_data = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
    return tokenized_data

# Define a custom Dataset class for labeled data
class SlovenianDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])

# Function to train model and calculate loss
def train_model_and_calculate_loss(model_name, tokenized_data):
    print("Training model: ", model_name)
    
    # Load training dataset
    labeled_dataset = SlovenianDataset(tokenized_data)
    labeled_dataloader = DataLoader(labeled_dataset, batch_size=16, shuffle=True)

    # Load a pre-trained model
    model = AutoModelForMaskedLM.from_pretrained(model_name)

    # Define Trainer
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        warmup_steps=500,
        logging_dir="./logs",
        logging_steps=100,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=labeled_dataloader,
        compute_metrics=None,
    )

    # Train the model
    trainer.train()

    # Calculate the final loss
    final_loss = trainer.evaluate()["loss"]
    
    return final_loss

# Iterate through 3 pretrained models
pretrained_models = ["bert-base-multilingual-uncased", "all-MiniLM-L12-v2", "hate_speech_slo"]
for model_name in pretrained_models:
    tokenized_data = tokenize_data(model_name, sentences)
    loss = train_model_and_calculate_loss(model_name, tokenized_data)
    print("Final loss for", model_name, ":", loss)
