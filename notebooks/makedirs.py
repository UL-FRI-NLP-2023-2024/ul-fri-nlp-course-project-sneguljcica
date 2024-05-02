import os

names = ["bert-base-multilingual-uncased", "all-MiniLM-L12-v2", "distiluse-base-multilingual-cased-v2", "distilbert-base-uncased"]


for name in names:
    os.makedirs(f"models/{name}", exist_ok=True)
    os.makedirs(f"models/tsdae-{name}", exist_ok=True)