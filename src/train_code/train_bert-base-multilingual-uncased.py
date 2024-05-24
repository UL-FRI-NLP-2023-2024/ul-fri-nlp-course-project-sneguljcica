from sentence_transformers import SentenceTransformer, models
import gpl
import torch
import os

# Print GPU Memory
# print("Total GPU Memory:", torch.cuda.get_device_properties(0).total_memory / 1e9, "GB")
# print("Available GPU Memory:", torch.cuda.memory_reserved(0) / 1e9, "GB")
# print("Allocated GPU Memory:", torch.cuda.memory_allocated(0) / 1e9, "GB")
# print("Free (cached) GPU Memory:", torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0) / 1e9, "GB")

# Paths
base_path = "/d/hpc/home/mm4195/model_final_100k/"
data_path = "/d/hpc/home/mm4195/data/final_data/100k_bert-base-multilingual-uncased/corpus.jsonl"
model_names = ["bert-base-multilingual-uncased"]

# Ensure data path exists
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Data file not found at {data_path}")


# Check GPU availability
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device('cpu')
    print("Using CPU")


# Function to train with GPL
def train_with_gpl(model_name):
    print(f"Training model: {model_name} with GPL")
    
    model_path = os.path.join(base_path, model_name)
    
    gpl.train(
        path_to_generated_data='/d/hpc/home/mm4195/data/final_data/100k_bert-base-multilingual-uncased',
        base_ckpt=model_path,
        batch_size_gpl=16,
        gpl_steps=70_000,
        output_dir=f'/d/hpc/home/mm4195/model_final_100k/{model_name}_gpl',
        generator='BeIR/query-gen-msmarco-t5-base-v1',
        retrievers=[
            'msmarco-distilbert-base-v3',
            'msmarco-MiniLM-L-6-v3'
        ],
        cross_encoder='cross-encoder/ms-marco-MiniLM-L-6-v2',
        qgen_prefix='qgen',
        do_evaluation=False
    )

# Train each model with GPL
for model_name in model_names:
    train_with_gpl(model_name)