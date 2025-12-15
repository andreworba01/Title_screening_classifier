"""
Configuration file for the title screening classifier.

Keeps all constants (paths, column names, hyperparameters) in one place 
"""

# -------------------------------------------------------------------
# Reproducibility
# -------------------------------------------------------------------
SEED = 42  # random seed used for NumPy, Python, and TensorFlow

# -------------------------------------------------------------------
# Data settings
# -------------------------------------------------------------------
DATA_PATH = "Data_title_screening.csv"  # CSV with Title + Inclusion
TEXT_COLUMN = "Title"                             # text field
LABEL_COLUMN = "Inclusion"                        # 1 = include, 0 = exclude

# -------------------------------------------------------------------
# Model and tokenizer
# -------------------------------------------------------------------
BERT_MODEL_NAME = "bert-base-uncased"  # huggingface model name

# -------------------------------------------------------------------
# Training hyperparameters
# -------------------------------------------------------------------
LEARNING_RATE = 3e-5
BATCH_SIZE = 16
EPOCHS = 10

# Path where the best model (based on val_loss) will be saved
BEST_MODEL_PATH = "best_bert_bilstm_attention_model.h5"

# -------------------------------------------------------------------
# Classification threshold
# -------------------------------------------------------------------
# Custom threshold chosen to maximize recall (minimize false negatives)
THRESHOLD = 0.22


