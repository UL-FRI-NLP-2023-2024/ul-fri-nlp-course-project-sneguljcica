import pandas as pd
import numpy as np

phrases = []

path = "/home/aljaz/FAKS/ul-fri-nlp-course-project-sneguljcica/data/raw/"
doc_names = ["AutoSentiNews_document-level_01.txt", "AutoSentiNews_document-level_02.txt"]

for doc_name in doc_names:

    documents = pd.read_csv(path + doc_name, sep='\t')
    documents = documents.iloc[:, 5]
    documents.columns = ['text']
    documents = documents.str.split('.' ).explode().reset_index(drop=True)
    phrases.extend(documents.to_list())

phrases = [phrase for phrase in phrases if len(phrase.split()) > 10]
pahrases = np.random.shuffle(phrases)
phrases = phrases[:100000]

df = pd.DataFrame(phrases, columns=['text'])
df.to_csv(path + '../processed/training_phrases100.csv', index=False, header=False)
