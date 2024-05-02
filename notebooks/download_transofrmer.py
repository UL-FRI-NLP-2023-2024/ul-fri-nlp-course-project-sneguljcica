from sentence_transformers import SentenceTransformer, models
import torch
torch.cuda.empty_cache() 

modelPath = "/home/aljaz/FAKS/ul-fri-nlp-course-project-sneguljcica/models/hate_speech_slo"


model_names = ["bert-base-multilingual-uncased", "all-MiniLM-L12-v2", "distiluse-base-multilingual-cased-v2", "distilbert-base-uncased"]


bert = models.Transformer('IMSyPP/hate_speech_slo')
pooling = models.Pooling(bert.get_word_embedding_dimension(), 'cls')
model = SentenceTransformer(modules=[bert, pooling])
# model = SentenceTransformer("bert-base-multilingual-uncased")
model.save(modelPath)

model = SentenceTransformer(modelPath)
print(model.encode("Dober dan svet!"))

# hate_speech_slo