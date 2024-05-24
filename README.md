# Natural language processing course 2023/24: `Unsupervised Domain Adaptation for Sentence Classification`

### Marko Možina, Peter Kosem, Aljaž Konec
==============================

## Project Description
This project sought to improve document representation in specialized domains by adapting sentence-transformer models, which, while effective, were not inherently tuned to specific fields. The focus was on investigating two advanced adaptation techniques: TSDAE (Transformer-based Denoising AutoEncoder) and GPL (Generative Pseudo Labeling). These methods aimed to refine the representation space, making it more sensitive and accurate within a given domain. We evaluated the effect of the adaptation on the classification results.

## Requirements
To run the code, you need to install the required dependencies. You can install them using conda and an `environment.yml` file.

```
git clone https://github.com/UL-FRI-NLP-2023-2024/ul-fri-nlp-course-project-sneguljcica.git
cd ul-fri-nlp-course-project-sneguljcica

conda env create -f environment.yml
conda activate nlp
```

## Reproducing the results

To reproduce the results, you need to run the following scripts while you have the `nlp` conda environment activated.

```
python eval_gpl_models.py
python eval_tsdae_models.py
```

## Results

The results of the experiments are stored in the `reports` directory. F1 scores and log loss of the models:

![TSDAE results](./reports/fig/tsdae_base.pdf)
![GPL results](./reports/fig/gpl_base.pdf)