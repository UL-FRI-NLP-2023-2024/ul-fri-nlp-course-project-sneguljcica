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

# Tokenize labeled data using a pre-trained tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-uncased")
tokenized_data = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")

# Define a custom Dataset class for labeled data
class SlovenianDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])

labeled_dataset = SlovenianDataset(tokenized_data)
labeled_dataloader = DataLoader(labeled_dataset, batch_size=16, shuffle=True)

# Load a pre-trained model
model_name = "bert-base-multilingual-uncased"
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
    train_dataset=labeled_dataset,
    compute_metrics=None,
)

# Train the model using GPL
trainer.train()

# Save the fine-tuned model
output_dir = "fine_tuned_model/"
os.makedirs(output_dir, exist_ok=True)
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
