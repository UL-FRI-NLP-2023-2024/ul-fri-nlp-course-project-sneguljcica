# Natural language processing course 2023/24: `Unsupervised Domain Adaptation for Sentence Classification`

### Marko Možina, Peter Kosem, Aljaž Konec
==============================

## Project Description
This project sought to improve document representation in specialized domains by adapting sentence-transformer models, which, while effective, were not inherently tuned to specific fields. The focus was on investigating two advanced adaptation techniques: TSDAE (Transformer-based Denoising AutoEncoder) and GPL (Generative Pseudo Labeling). These methods aimed to refine the representation space, making it more sensitive and accurate within a given domain. We evaluated the effect of the adaptation on the classification results.

## Requirements
To run the code, you need to install the required dependencies. You can install them using pip and a requirements.txt file.

```
git clone https://github.com/UL-FRI-NLP-2023-2024/ul-fri-nlp-course-project-sneguljcica.git
cd ul-fri-nlp-course-project-sneguljcica

python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

pip install -r requirements.txt
```

