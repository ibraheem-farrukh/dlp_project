import pandas as pd
import re
import unicodedata
from sklearn.model_selection import train_test_split
from collections import Counter
import pickle

# Load dataset
df = pd.read_csv("data/output_ur.csv")

# Assume poem text is in 'content' column
poem_col = 'content'

def clean_text(text):
    """
    Clean and normalize Urdu text
    """
    if pd.isna(text):
        return ""

    # Convert to string
    text = str(text)

    # Normalize Unicode (handle different representations of same characters)
    text = unicodedata.normalize('NFC', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())

    # Remove control characters but keep Urdu characters
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)

    # Handle common Urdu special characters - keep Urdu script and basic punctuation
    # Remove non-Urdu characters except spaces and basic punctuation
    text = re.sub(r'[^\u0600-\u06FF\s\.,!?؛:۔]', '', text)

    return text.strip()

def normalize_urdu_text(text):
    """
    Additional Urdu-specific normalization
    """
    if not text:
        return ""

    # Handle Urdu specific character variations
    # This is a basic implementation - for more advanced normalization,
    # consider using urduhack library

    # Normalize alef variants
    text = re.sub(r'[اأإآ]', 'ا', text)

    # Normalize yeh variants
    text = re.sub(r'[يى]', 'ی', text)

    # Normalize heh variants
    text = re.sub(r'[هة]', 'ہ', text)

    return text

def tokenize_text(text):
    """
    Tokenize Urdu text
    """
    if not text:
        return []

    # Simple tokenization by whitespace
    # For more advanced tokenization, consider using urduhack or hazm
    tokens = text.split()

    return tokens

def preprocess_data(df, poem_col='content'):
    """
    Main preprocessing pipeline
    """
    print("Starting data preprocessing...")

    # Clean text
    print("Cleaning text...")
    df['cleaned_text'] = df[poem_col].apply(clean_text)

    # Normalize Urdu text
    print("Normalizing Urdu text...")
    df['normalized_text'] = df['cleaned_text'].apply(normalize_urdu_text)

    # Filter empty lines
    print("Filtering empty texts...")
    df = df[df['normalized_text'].str.len() > 0].copy()

    # Tokenize
    print("Tokenizing text...")
    df['tokens'] = df['normalized_text'].apply(tokenize_text)

    # Filter out empty token lists
    df = df[df['tokens'].str.len() > 0].copy()

    print(f"Preprocessing complete. Remaining samples: {len(df)}")

    return df

def build_vocabulary(tokens_list, min_freq=1):
    """
    Build vocabulary from tokenized texts
    """
    print("Building vocabulary...")

    # Flatten all tokens
    all_tokens = [token for tokens in tokens_list for token in tokens]

    # Count frequencies
    token_counts = Counter(all_tokens)

    # Filter by minimum frequency
    vocab = {token: count for token, count in token_counts.items() if count >= min_freq}

    # Sort by frequency (most common first)
    vocab = dict(sorted(vocab.items(), key=lambda x: x[1], reverse=True))

    # Add special tokens
    special_tokens = {
        '<PAD>': len(vocab) + 1,  # Padding token
        '<UNK>': len(vocab) + 2,  # Unknown token
        '<SOS>': len(vocab) + 3,  # Start of sequence
        '<EOS>': len(vocab) + 4   # End of sequence
    }

    # Create token to index mapping
    token_to_idx = {token: idx for idx, token in enumerate(vocab.keys(), start=1)}
    token_to_idx.update(special_tokens)

    # Create index to token mapping
    idx_to_token = {idx: token for token, idx in token_to_idx.items()}

    print(f"Vocabulary size: {len(token_to_idx)}")

    return token_to_idx, idx_to_token, vocab

def prepare_sequences(tokens_list, token_to_idx, max_length=None):
    """
    Convert tokens to numerical sequences
    """
    print("Preparing sequences...")

    sequences = []
    for tokens in tokens_list:
        # Convert tokens to indices
        seq = [token_to_idx.get(token, token_to_idx['<UNK>']) for token in tokens]

        # Add start and end tokens
        seq = [token_to_idx['<SOS>']] + seq + [token_to_idx['<EOS>']]

        sequences.append(seq)

    # Pad sequences to same length if max_length specified
    if max_length:
        for seq in sequences:
            if len(seq) < max_length:
                seq.extend([token_to_idx['<PAD>']] * (max_length - len(seq)))
            elif len(seq) > max_length:
                seq = seq[:max_length]

    return sequences

def split_data(df, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_state=42):
    """
    Split data into train, validation, and test sets
    """
    assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1.0"

    print("Splitting data...")

    # First split: train and temp (val+test)
    train_df, temp_df = train_test_split(
        df,
        train_size=train_ratio,
        random_state=random_state,
        shuffle=True
    )

    # Second split: val and test from temp
    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_ratio_adjusted,
        random_state=random_state,
        shuffle=True
    )

    print(f"Train set: {len(train_df)} samples")
    print(f"Validation set: {len(val_df)} samples")
    print(f"Test set: {len(test_df)} samples")

    return train_df, val_df, test_df

def save_processed_data(data_dict, filename):
    """
    Save processed data to pickle file
    """
    with open(filename, 'wb') as f:
        pickle.dump(data_dict, f)
    print(f"Data saved to {filename}")

# Main execution
if __name__ == "__main__":
    # Preprocess data
    processed_df = preprocess_data(df, poem_col)

    # Build vocabulary
    token_to_idx, idx_to_token, vocab = build_vocabulary(processed_df['tokens'])

    # Prepare sequences (without padding for now)
    sequences = prepare_sequences(processed_df['tokens'], token_to_idx)

    # Add sequences to dataframe
    processed_df['sequences'] = sequences

    # Split data
    train_df, val_df, test_df = split_data(processed_df)

    # Prepare final datasets
    train_sequences = train_df['sequences'].tolist()
    val_sequences = val_df['sequences'].tolist()
    test_sequences = test_df['sequences'].tolist()

    # Save processed data
    processed_data = {
        'train_sequences': train_sequences,
        'val_sequences': val_sequences,
        'test_sequences': test_sequences,
        'token_to_idx': token_to_idx,
        'idx_to_token': idx_to_token,
        'vocab': vocab,
        'original_data': {
            'train': train_df,
            'val': val_df,
            'test': test_df
        }
    }

    save_processed_data(processed_data, 'data/processed_data.pkl')

    print("\nPreprocessing completed successfully!")
    print(f"Total vocabulary size: {len(token_to_idx)}")
    print(f"Train samples: {len(train_sequences)}")
    print(f"Validation samples: {len(val_sequences)}")
    print(f"Test samples: {len(test_sequences)}")
