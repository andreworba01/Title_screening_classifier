"""
Model and utility functions for the title screening classifier.

This module defines:
- Seed setting function (for reproducibility)
- Tokenizer loader
- Text encoding helper
- Max-length computation
- BERT + BiLSTM + Multi-Head Attention architecture
- Evaluation helper (accuracy, precision, recall, confusion matrix)
"""

import os
import random
import numpy as np
import tensorflow as tf

from transformers import BertTokenizer, TFBertModel
from tensorflow.keras.layers import (
    Input, GlobalAveragePooling1D, Dropout, Dense, Bidirectional, LSTM
)
from tensorflow_addons.losses import SigmoidFocalCrossEntropy
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

from title_screening_config import (
    SEED, BERT_MODEL_NAME, LEARNING_RATE, THRESHOLD
)

# -------------------------------------------------------------------
# Reproducibility helpers
# -------------------------------------------------------------------
def set_seeds():
    """
    Set random seeds for NumPy, Python, and TensorFlow to make results
    as reproducible as possible.
    """
    np.random.seed(SEED)
    random.seed(SEED)
    tf.random.set_seed(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)


# -------------------------------------------------------------------
# Tokenizer utilities
# -------------------------------------------------------------------
def get_tokenizer():
    """
    Load the BERT tokenizer used throughout the project.
    """
    return BertTokenizer.from_pretrained(BERT_MODEL_NAME)


def compute_max_len(sentences, tokenizer):
    """
    Compute token lengths for all training sentences and return:
    - max length
    - mean length
    - median length

    This helps set the max sequence length for the model.
    """
    lengths = []
    for s in sentences:
        ids = tokenizer.encode(str(s), add_special_tokens=True)
        lengths.append(len(ids))

    max_len = max(lengths)
    mean_len = float(np.mean(lengths))
    median_len = float(np.median(lengths))

    return max_len, mean_len, median_len


def encode_texts(texts, tokenizer, max_len):
    """
    Tokenize a list/array of texts using the BERT tokenizer.

    Returns a dictionary of:
    - input_ids
    - attention_mask

    Both are TensorFlow tensors ready to be fed into the model.
    """
    texts = [str(t) for t in texts]
    encodings = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_tensors="tf",
    )
    return encodings


# -------------------------------------------------------------------
# Model architecture
# -------------------------------------------------------------------
def create_model_with_bilstm_and_attention(max_seq_len: int) -> tf.keras.Model:
    """
    Build and compile the BERT + BiLSTM + Multi-Head Attention model.

    Architecture:
    - BERT embeddings (TFBertModel)
    - BiLSTM(64) + Dropout
    - Multi-Head Attention
    - BiLSTM(32) + Dropout
    - GlobalAveragePooling1D
    - Dense(64) + Dropout
    - Dense(32) + Dropout
    - Output: Dense(1, sigmoid)

    Uses Sigmoid Focal Cross-Entropy as the loss function to handle
    class imbalance and emphasize hard examples.
    """

    # Input layers: token IDs + attention mask
    input_ids = Input(shape=(max_seq_len,), dtype=tf.int32, name="input_ids")
    input_mask = Input(shape=(max_seq_len,), dtype=tf.int32, name="attention_mask")

    # Pretrained BERT backbone
    bert_model = TFBertModel.from_pretrained(BERT_MODEL_NAME)
    bert_outputs = bert_model(input_ids, attention_mask=input_mask)

    # Last hidden states for each token
    x = bert_outputs.last_hidden_state  # shape: (batch, seq_len, hidden_dim)

    # First BiLSTM layer
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = Dropout(0.3)(x)

    # Multi-head self-attention layer
    attention_layer = tf.keras.layers.MultiHeadAttention(num_heads=2, key_dim=64)
    x = attention_layer(x, x)

    # Second BiLSTM layer
    x = Bidirectional(LSTM(32, return_sequences=True))(x)
    x = Dropout(0.3)(x)

    # Pool across sequence dimension
    x = GlobalAveragePooling1D()(x)

    # Dense layers for classification
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.4)(x)
    x = Dense(32, activation="relu")(x)
    x = Dropout(0.3)(x)

    # Final output: probability of Inclusion = 1
    output = Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=[input_ids, input_mask], outputs=output)

    # Use focal loss to emphasize harder (misclassified) examples
    loss_fn = SigmoidFocalCrossEntropy(alpha=0.55, gamma=1.2)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=loss_fn,
        metrics=[
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="AUC"),
        ],
    )

    return model


# -------------------------------------------------------------------
# Evaluation helper
# -------------------------------------------------------------------
def evaluate_model(model, encodings, labels, threshold: float = THRESHOLD):
    """
    Evaluate a trained model using a custom classification threshold.

    Parameters
    ----------
    model : tf.keras.Model
        Trained classifier.
    encodings : dict
        Dictionary with 'input_ids' and 'attention_mask' tensors.
    labels : array-like
        True binary labels (0/1).
    threshold : float
        Probability threshold used to assign class 1.

    Returns
    -------
    accuracy, precision, recall, confusion_matrix
    """
    probs = model.predict(
        {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
        },
        verbose=0,
    )

    preds = (probs >= threshold).astype(int)

    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    cm = confusion_matrix(labels, preds)

    return acc, prec, rec, cm
