"""
Train and evaluate the title screening classifier for the MNP systematic review.

Workflow:
1. Set seeds for reproducibility.
2. Load labeled title dataset (Title + Inclusion).
3. Check for duplicate rows and remove them (with reporting).
4. Show class distribution (0 = exclude, 1 = include).
5. Split into train / validation / test sets (stratified).
6. Compute max BERT token length from training titles.
7. Tokenize titles using BERT tokenizer.
8. Build and train BERT + BiLSTM + Attention model.
9. Save best model based on validation loss.
10. Evaluate model on train, validation, test, and full dataset.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from title_screening_config import (
    DATA_PATH, TEXT_COLUMN, LABEL_COLUMN,
    BEST_MODEL_PATH, EPOCHS, BATCH_SIZE, THRESHOLD
)
from title_screening_model import (
    set_seeds, get_tokenizer, compute_max_len, encode_texts,
    create_model_with_bilstm_and_attention, evaluate_model
)


def main():
    # -----------------------------------------------------------
    # 1. Reproducibility
    # -----------------------------------------------------------
    set_seeds()

    # -----------------------------------------------------------
    # 2. Load data
    # -----------------------------------------------------------
    df = pd.read_csv(DATA_PATH)

    print("\n--- Initial Data Summary ---")
    print("Initial shape:", df.shape)

    # Basic validation of expected columns
    assert TEXT_COLUMN in df.columns, f"Column '{TEXT_COLUMN}' not found."
    assert LABEL_COLUMN in df.columns, f"Column '{LABEL_COLUMN}' not found."

    # Ensure label is integer and title is string
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int)
    df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str)

    # -----------------------------------------------------------
    # 3. Check and remove duplicate rows
    # -----------------------------------------------------------
    duplicate_rows = df[df.duplicated()]  # duplicated across all columns

    print("\n--- Duplicate Check ---")
    print(f"Number of duplicated rows: {duplicate_rows.shape[0]}")

    if not duplicate_rows.empty:
        print("Example duplicated rows:")
        print(duplicate_rows.head())
    else:
        print("No duplicated rows found.")

    # Remove all duplicates, keep first occurrence
    df = df.drop_duplicates()

    print("\nAfter duplicate removal, shape:", df.shape)

    # -----------------------------------------------------------
    # 4. Class distribution (after cleaning)
    # -----------------------------------------------------------
    print("\n--- Class Distribution (0=Exclude, 1=Include) ---")
    print(df[LABEL_COLUMN].value_counts())

    # Extract NumPy arrays
    sentences_all = df[TEXT_COLUMN].values
    labels_all = df[LABEL_COLUMN].values

    # -----------------------------------------------------------
    # 5. Train / validation / test split (stratified)
    # -----------------------------------------------------------
    sentences_train_all, sentences_val, labels_train_all, labels_val = train_test_split(
        sentences_all,
        labels_all,
        test_size=0.20,
        random_state=142,
        stratify=labels_all,
    )

    sentences_train, sentences_test, labels_train, labels_test = train_test_split(
        sentences_train_all,
        labels_train_all,
        test_size=0.15,
        random_state=1542,
        stratify=labels_train_all,
    )

    print("\n--- Dataset Sizes ---")
    print("Train:", len(sentences_train))
    print("Validation:", len(sentences_val))
    print("Test:", len(sentences_test))

    # -----------------------------------------------------------
    # 6. Tokenizer and max sequence length
    # -----------------------------------------------------------
    tokenizer = get_tokenizer()
    max_len, mean_len, median_len = compute_max_len(sentences_train, tokenizer)

    print("\n--- Token Length Statistics (Training Set) ---")
    print(f"Max:    {max_len}")
    print(f"Mean:   {mean_len:.2f}")
    print(f"Median: {median_len:.2f}")

    # -----------------------------------------------------------
    # 7. Encode texts for all splits
    # -----------------------------------------------------------
    train_enc = encode_texts(sentences_train, tokenizer, max_len)
    val_enc = encode_texts(sentences_val, tokenizer, max_len)
    test_enc = encode_texts(sentences_test, tokenizer, max_len)
    all_enc = encode_texts(sentences_all, tokenizer, max_len)

    # -----------------------------------------------------------
    # 8. Build model
    # -----------------------------------------------------------
    model = create_model_with_bilstm_and_attention(max_len)
    model.summary()

    # -----------------------------------------------------------
    # 9. Callbacks for training
    # -----------------------------------------------------------
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            BEST_MODEL_PATH,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    # -----------------------------------------------------------
    # 10. Train the model
    # -----------------------------------------------------------
    history = model.fit(
        {
            "input_ids": train_enc["input_ids"],
            "attention_mask": train_enc["attention_mask"],
        },
        labels_train,
        validation_data=(
            {
                "input_ids": val_enc["input_ids"],
                "attention_mask": val_enc["attention_mask"],
            },
            labels_val,
        ),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
    )

    print(f"\nTraining complete. Best model saved to: {BEST_MODEL_PATH}")

    # -----------------------------------------------------------
    # 11. Evaluation on all splits
    # -----------------------------------------------------------
    print(f"\n--- Evaluation at threshold = {THRESHOLD} ---")

    datasets = {
        "Train": (train_enc, labels_train),
        "Validation": (val_enc, labels_val),
        "Test": (test_enc, labels_test),
        "Full Dataset": (all_enc, labels_all),
    }

    for name, (enc, lab) in datasets.items():
        acc, prec, rec, cm = evaluate_model(model, enc, lab)
        print(f"\n{name} set:")
        print("Confusion matrix:\n", cm)
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")


if __name__ == "__main__":
    main()
