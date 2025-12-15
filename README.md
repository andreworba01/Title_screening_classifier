
# Title Screening Classifier (LLM-Based Systematic Review Tool)

This repository contains the data and code used to develop the large-language-model (LLM) classifier that screened 41,784 scientific titles as part of a systematic review on mammalian nano- and microplastic (MNP) toxicity. The classifier was designed to maximize recall and ensure that no potentially relevant studies were excluded during the title-screening stage.

---

## Contents

- Data_title_screening.csv – Labeled dataset of titles used for model training and evaluation

- SR_BERT_NPS.ipynb – Reproducible notebook containing preprocessing, model training, evaluation, and inference

- Model architecture – Fine-tuned BERT + BiLSTM + Multi-Head Attention neural network

- Utilities – Tokenization scripts, thresholding logic, and evaluation metrics

--- 

### Model Summary

- Backbone: BERT-base-uncased

- Additional layers: BiLSTM + Multi-Head Attention + Dense layers

- Optimized for 100% recall to avoid false negatives

- Final threshold: 0.22, enabling maximum sensitivity

---

### Data description 
It contains manually labeled scientific titles used to train and evaluate the classifier.
Each row consists of:

Title – The scientific article title

Inclusion – Relevance label

1 = Include (title meets PECOS criteria for potential relevance)

0 = Exclude (title does not meet criteria)

This dataset was used to fine-tune the model and calibrate the decision threshold.

Full information on the PECOS criteria can be found in the published protocol at: https://www.crd.york.ac.uk/PROSPERO/view/CRD420250643483 
