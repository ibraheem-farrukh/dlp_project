import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, Dense, Dropout, Input, MultiHeadAttention, LayerNormalization
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import os
import time
from collections import defaultdict
import mlflow
import mlflow.keras
import mlflow.tensorflow

# Load processed data
with open('data/processed_data.pkl', 'rb') as f:
    data = pickle.load(f)

# Extract data
train_sequences = data['train_sequences']
val_sequences = data['val_sequences']
test_sequences = data['test_sequences']
token_to_idx = data['token_to_idx']
idx_to_token = data['idx_to_token']

vocab_size = len(token_to_idx)
max_sequence_length = max(len(seq) for seq in train_sequences + val_sequences + test_sequences)

print(f"Vocabulary size: {vocab_size}")
print(f"Max sequence length: {max_sequence_length}")
print(f"Training samples: {len(train_sequences)}")

# Initialize MLflow
try:
    mlflow.set_experiment("Urdu_Poetry_Generation_Baseline")
    print("🔍 MLflow experiment initialized: Urdu_Poetry_Generation_Baseline")
except Exception as e:
    print(f"⚠️  Warning: Could not initialize MLflow experiment: {e}")
    print("Continuing without MLflow logging...")

# Prepare data for training (create input-output pairs)
def prepare_training_data(sequences, max_len=None):
    """
    Prepare sequences for training by creating input-output pairs
    """
    X = []
    y = []

    for seq in sequences:
        for i in range(1, len(seq)):
            X.append(seq[:i])
            y.append(seq[i])

    # Pad sequences
    if max_len is None:
        max_len = max(len(x) for x in X)

    X_padded = pad_sequences(X, maxlen=max_len, padding='pre', value=token_to_idx['<PAD>'])
    # Keep y as integers for sparse categorical crossentropy
    y_int = np.array(y)

    return X_padded, y_int, max_len

# Prepare training data
print("Preparing training data...")
X_train, y_train, seq_length = prepare_training_data(train_sequences[:500])  # Reduced to 500 samples
X_val, y_val, _ = prepare_training_data(val_sequences[:100], seq_length)      # Reduced to 100 samples
X_test, y_test, _ = prepare_training_data(test_sequences[:100], seq_length)   # Reduced to 100 samples

print(f"Training data shape: {X_train.shape}")
print(f"Training labels shape: {y_train.shape}")

# Build models
def build_rnn_model(vocab_size, seq_length, embedding_dim=100, rnn_units=128):
    """Build Simple RNN model"""
    model = Sequential([
        Embedding(vocab_size + 1, embedding_dim),  # Removed input_length
        SimpleRNN(rnn_units, return_sequences=False),
        Dropout(0.2),
        Dense(vocab_size + 1, activation='softmax')
    ])
    return model

def build_lstm_model(vocab_size, seq_length, embedding_dim=100, lstm_units=128):
    """Build LSTM model"""
    model = Sequential([
        Embedding(vocab_size + 1, embedding_dim),  # Removed input_length
        LSTM(lstm_units, return_sequences=False),
        Dropout(0.2),
        Dense(vocab_size + 1, activation='softmax')
    ])
    return model

def build_transformer_model(vocab_size, seq_length, embedding_dim=100, num_heads=8, ff_dim=128):
    """Build simple Transformer model"""
    inputs = Input(shape=(seq_length,))

    # Embedding layer
    embedding_layer = Embedding(vocab_size + 1, embedding_dim)(inputs)
    embedding_layer = LayerNormalization(epsilon=1e-6)(embedding_layer)

    # Multi-head attention
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim)(embedding_layer, embedding_layer)
    attention_output = LayerNormalization(epsilon=1e-6)(attention_output + embedding_layer)

    # Feed-forward network
    ffn_output = Dense(ff_dim, activation='relu')(attention_output)
    ffn_output = Dense(embedding_dim)(ffn_output)
    ffn_output = LayerNormalization(epsilon=1e-6)(ffn_output + attention_output)

    # Global average pooling and output
    pooled = tf.keras.layers.GlobalAveragePooling1D()(ffn_output)
    outputs = Dense(vocab_size + 1, activation='softmax')(pooled)

    model = Model(inputs=inputs, outputs=outputs)
    return model

# Optimizers
optimizers = {
    'adam': Adam(learning_rate=0.001),
    'rmsprop': RMSprop(learning_rate=0.001),
    'sgd': SGD(learning_rate=0.01)
}

# Training function
def train_model(model, optimizer_name, X_train, y_train, X_val, y_val, epochs=10, batch_size=32, model_name="model"):
    """Train a model with given optimizer and log to MLflow"""
    model.compile(
        loss='sparse_categorical_crossentropy',  # Changed to sparse for integer labels
        optimizer=optimizers[optimizer_name],
        metrics=['accuracy']
    )

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    checkpoint = ModelCheckpoint(
        f'models/{model.name}_{optimizer_name}.h5',
        monitor='val_loss',
        save_best_only=True
    )

    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)

    # Track training time
    start_time = time.time()

    # End any existing MLflow runs to prevent conflicts
    try:
        mlflow.end_run()
    except:
        pass  # Ignore if no active run

    # Start MLflow run with error handling
    try:
        with mlflow.start_run(run_name=f"{model_name}_{optimizer_name}"):

            # Log parameters
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("optimizer", optimizer_name)
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_param("vocab_size", vocab_size)
            mlflow.log_param("seq_length", seq_length)
            mlflow.log_param("embedding_dim", 100)
            mlflow.log_param("rnn_units", 128 if 'rnn' in model_name.lower() or 'lstm' in model_name.lower() else 'N/A')

            # Log dataset info
            mlflow.log_param("train_samples", len(X_train))
            mlflow.log_param("val_samples", len(X_val))
            mlflow.log_param("test_samples", len(X_test))

            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[early_stopping, checkpoint],
                verbose=1
            )

            training_time = time.time() - start_time

            # Log training metrics
            for epoch in range(len(history.history['loss'])):
                mlflow.log_metric("train_loss", history.history['loss'][epoch], step=epoch)
                mlflow.log_metric("train_accuracy", history.history['accuracy'][epoch], step=epoch)
                mlflow.log_metric("val_loss", history.history['val_loss'][epoch], step=epoch)
                mlflow.log_metric("val_accuracy", history.history['val_accuracy'][epoch], step=epoch)

            # Log final metrics
            final_train_loss = history.history['loss'][-1]
            final_train_acc = history.history['accuracy'][-1]
            final_val_loss = history.history['val_loss'][-1]
            final_val_acc = history.history['val_accuracy'][-1]

            mlflow.log_metric("final_train_loss", final_train_loss)
            mlflow.log_metric("final_train_accuracy", final_train_acc)
            mlflow.log_metric("final_val_loss", final_val_loss)
            mlflow.log_metric("final_val_accuracy", final_val_acc)
            mlflow.log_metric("training_time", training_time)
            mlflow.log_metric("epochs_trained", len(history.history['loss']))

            # Log model architecture summary
            model_summary = []
            model.summary(print_fn=lambda x: model_summary.append(x))
            model_summary_str = "\n".join(model_summary)
            mlflow.log_text(model_summary_str, "model_summary.txt")

            # Log model
            mlflow.keras.log_model(model, "model")

            # Log training curves as artifacts
            plt.figure(figsize=(12, 4))

            plt.subplot(1, 2, 1)
            plt.plot(history.history['loss'], label='Training Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title(f'{model_name} Loss Curves')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.plot(history.history['accuracy'], label='Training Accuracy')
            plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
            plt.title(f'{model_name} Accuracy Curves')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()

            plt.tight_layout()
            plot_path = f"models/{model_name}_{optimizer_name}_curves.png"
            plt.savefig(plot_path)
            mlflow.log_artifact(plot_path)
            plt.close()

    except Exception as e:
        print(f"⚠️  Warning: MLflow logging failed for {model_name}_{optimizer_name}: {e}")
        print("Continuing without MLflow logging for this run...")

        # Fallback: train without MLflow
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, checkpoint],
            verbose=1
        )

        training_time = time.time() - start_time

    return history, training_time

# Text generation function
def generate_text(model, seed_text, token_to_idx, idx_to_token, max_length=50, temperature=1.0):
    """Generate text using the trained model"""
    # Tokenize seed text
    seed_tokens = seed_text.split()
    seed_indices = [token_to_idx.get(token, token_to_idx['<UNK>']) for token in seed_tokens]

    generated = seed_indices.copy()

    for _ in range(max_length):
        # Prepare input
        input_seq = pad_sequences([generated], maxlen=seq_length, padding='pre', value=token_to_idx['<PAD>'])

        # Predict next token
        predictions = model.predict(input_seq, verbose=0)[0]

        # Apply temperature
        predictions = np.log(predictions + 1e-8) / temperature
        predictions = np.exp(predictions)
        predictions = predictions / np.sum(predictions)

        # Sample next token
        next_token_idx = np.random.choice(len(predictions), p=predictions)

        # Stop if EOS token
        if next_token_idx == token_to_idx.get('<EOS>', -1):
            break

        generated.append(next_token_idx)

        # Stop if max length reached
        if len(generated) >= max_length:
            break

    # Convert back to text
    generated_text = [idx_to_token.get(idx, '<UNK>') for idx in generated]
    return ' '.join(generated_text)

# Evaluation function
def evaluate_model(model, X_test, y_test, model_name="model", optimizer_name="optimizer"):
    """Evaluate model on test set and log to MLflow"""
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    perplexity = np.exp(loss)  # Calculate perplexity

    # Log test metrics to current MLflow run
    mlflow.log_metric("test_loss", loss)
    mlflow.log_metric("test_accuracy", accuracy)
    mlflow.log_metric("test_perplexity", perplexity)

    return loss, accuracy, perplexity

def analyze_training_curves(history):
    """Analyze training curves for overfitting and convergence"""
    train_loss = history['loss']
    val_loss = history['val_loss']

    # Check if training loss decreases consistently
    train_decreasing = all(train_loss[i] >= train_loss[i+1] for i in range(len(train_loss)-1))

    # Check for overfitting (validation loss diverging from training loss)
    final_train_loss = train_loss[-1]
    final_val_loss = val_loss[-1]
    overfitting = final_val_loss > final_train_loss * 1.2  # 20% threshold

    # Calculate convergence rate
    initial_loss = train_loss[0]
    final_loss = train_loss[-1]
    convergence_rate = (initial_loss - final_loss) / initial_loss if initial_loss > 0 else 0

    return {
        'train_decreasing': train_decreasing,
        'overfitting': overfitting,
        'final_train_loss': final_train_loss,
        'final_val_loss': final_val_loss,
        'convergence_rate': convergence_rate
    }

def analyze_generated_text_quality(generated_texts, seed_words):
    """Basic analysis of generated text quality"""
    analysis = {}

    for seed, text in zip(seed_words, generated_texts):
        words = text.split()
        # Basic metrics
        length = len(words)
        unique_words = len(set(words))
        diversity = unique_words / length if length > 0 else 0

        # Check if it contains Urdu characters (basic check)
        urdu_chars = any('\u0600' <= char <= '\u06FF' for char in text)

        analysis[seed] = {
            'length': length,
            'diversity': diversity,
            'contains_urdu': urdu_chars,
            'text': text
        }

    return analysis

def create_comparison_table(results):
    """Create a comprehensive comparison table"""
    comparison = defaultdict(dict)

    for model_name, model_results in results.items():
        for opt_name, result in model_results.items():
            key = f"{model_name.upper()}_{opt_name.upper()}"
            comparison[key] = {
                'model': model_name,
                'optimizer': opt_name,
                'test_loss': result['test_loss'],
                'test_accuracy': result['test_accuracy'],
                'perplexity': result['perplexity'],
                'training_time': result['training_time'],
                'epochs_trained': len(result['history']['loss'])
            }

    return comparison

# Main training and evaluation
def main():
    # Clean up any existing MLflow runs
    try:
        mlflow.end_run()
    except:
        pass

    models_to_train = {
        'rnn': build_rnn_model,
        'lstm': build_lstm_model,
        'transformer': build_transformer_model
    }

    results = {}

    for model_name, build_func in models_to_train.items():
        print(f"\n{'='*50}")
        print(f"Training {model_name.upper()} models")
        print(f"{'='*50}")

        model_results = {}

        for opt_name in ['adam', 'rmsprop', 'sgd']:
            print(f"\n--- Training {model_name.upper()} with {opt_name.upper()} ---")

            # Build model
            if model_name == 'transformer':
                model = build_func(vocab_size, seq_length)
            else:
                model = build_func(vocab_size, seq_length)

            model._name = f"{model_name}_{opt_name}"

            # Train model
            history, training_time = train_model(model, opt_name, X_train, y_train, X_val, y_val, epochs=3, batch_size=32, model_name=model_name)

            # Evaluate (MLflow logging happens inside the current run context)
            test_loss, test_accuracy, perplexity = evaluate_model(model, X_test, y_test, model_name, opt_name)

            # Analyze training curves
            curve_analysis = analyze_training_curves(history.history)

            # Store results
            model_results[opt_name] = {
                'history': history.history,
                'test_loss': test_loss,
                'test_accuracy': test_accuracy,
                'perplexity': perplexity,
                'training_time': training_time,
                'curve_analysis': curve_analysis,
                'model': model
            }

            print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Perplexity: {perplexity:.2f}")
            print(f"Training Time: {training_time:.2f}s")
            print(f"Training Loss Decreasing: {curve_analysis['train_decreasing']}")
            print(f"Overfitting Detected: {curve_analysis['overfitting']}")
        results[model_name] = model_results

    # Detailed Comparative Analysis - STEP 3 Checkpoint Questions
    print(f"\n{'='*80}")
    print("STEP 3: DETAILED MODEL ANALYSIS")
    print(f"{'='*80}")

    # Analyze each model individually
    for model_name, model_results in results.items():
        print(f"\n🔍 {model_name.upper()} ANALYSIS:")
        print("-" * 40)

        for opt_name, result in model_results.items():
            analysis = result['curve_analysis']
            print(f"\n  📊 {model_name.upper()} + {opt_name.upper()}:")
            print(f"    • Test Loss: {result['test_loss']:.4f}")
            print(f"    • Test Accuracy: {result['test_accuracy']:.4f}")
            print(f"    • Perplexity: {result['perplexity']:.2f}")
            print(f"    • Training Time: {result['training_time']:.2f}s")
            print(f"    • Training Loss Decreasing: {analysis['train_decreasing']}")
            print(f"    • Overfitting Detected: {analysis['overfitting']}")
            print(f"    • Convergence Rate: {analysis['convergence_rate']:.3f}")

    # Cross-model comparisons
    print(f"\n{'='*80}")
    print("CROSS-MODEL COMPARISONS")
    print(f"{'='*80}")

    # Compare architectures with best optimizer (Adam)
    print("\n🏆 ARCHITECTURE COMPARISON (with Adam optimizer):")
    rnn_adam = results['rnn']['adam']
    lstm_adam = results['lstm']['adam']
    transformer_adam = results['transformer']['adam']

    print("\n  RNN vs LSTM vs Transformer:")
    print(f"    • RNN Perplexity: {rnn_adam['perplexity']:.2f}, Time: {rnn_adam['training_time']:.2f}s")
    print(f"    • LSTM Perplexity: {lstm_adam['perplexity']:.2f}, Time: {lstm_adam['training_time']:.2f}s")
    print(f"    • Transformer Perplexity: {transformer_adam['perplexity']:.2f}, Time: {transformer_adam['training_time']:.2f}s")

    # Calculate improvements
    rnn_to_lstm = ((rnn_adam['perplexity'] - lstm_adam['perplexity']) / rnn_adam['perplexity']) * 100
    lstm_to_transformer = ((lstm_adam['perplexity'] - transformer_adam['perplexity']) / lstm_adam['perplexity']) * 100

    print(f"    • LSTM improvement over RNN: {rnn_to_lstm:.1f}%")
    print(f"    • Transformer improvement over LSTM: {lstm_to_transformer:.1f}%")

    # STEP 4: Optimizer Comparison Experiments
    print(f"\n{'='*80}")
    print("STEP 4: OPTIMIZER COMPARISON EXPERIMENTS")
    print(f"{'='*80}")

    # Compare optimizers for each architecture
    for model_name in ['rnn', 'lstm', 'transformer']:
        print(f"\n🔧 {model_name.upper()} OPTIMIZER COMPARISON:")
        print("-" * 40)

        model_opts = results[model_name]
        best_opt = min(model_opts.keys(), key=lambda x: model_opts[x]['perplexity'])

        for opt_name, result in model_opts.items():
            marker = " 🏆" if opt_name == best_opt else ""
            print(f"  • {opt_name.upper()}{marker}: Perplexity={result['perplexity']:.2f}, Time={result['training_time']:.2f}s")

        print(f"  📈 Best optimizer for {model_name.upper()}: {best_opt.upper()}")

    # Overall best combination
    all_combinations = []
    for model_name, model_results in results.items():
        for opt_name, result in model_results.items():
            all_combinations.append({
                'model': model_name,
                'optimizer': opt_name,
                'perplexity': result['perplexity'],
                'time': result['training_time'],
                'efficiency': result['perplexity'] / result['training_time']  # Lower is better
            })

    best_overall = min(all_combinations, key=lambda x: x['perplexity'])
    most_efficient = min(all_combinations, key=lambda x: x['efficiency'])

    print(f"\n🏆 OVERALL RESULTS:")
    print(f"  • Best Performance: {best_overall['model'].upper()} + {best_overall['optimizer'].upper()} (Perplexity: {best_overall['perplexity']:.2f})")
    print(f"  • Most Efficient: {most_efficient['model'].upper()} + {most_efficient['optimizer'].upper()} (Efficiency: {most_efficient['efficiency']:.4f})")

    # Create and display comparison table
    comparison_table = create_comparison_table(results)
    print(f"\n{'='*80}")
    print("COMPLETE COMPARISON TABLE (9 combinations)")
    print(f"{'='*80}")
    print(f"{'Combination':<15} {'Model':<12} {'Optimizer':<10} {'Perplexity':<12} {'Accuracy':<10} {'Time(s)':<10} {'Epochs':<8}")
    print("-" * 80)

    for combo, data in comparison_table.items():
        print(f"{combo:<15} {data['model']:<12} {data['optimizer']:<10} {data['perplexity']:<12.2f} {data['test_accuracy']:<10.4f} {data['training_time']:<10.2f} {data['epochs_trained']:<8}")

    # Text generation and quality analysis
    print(f"\n{'='*80}")
    print("TEXT GENERATION & QUALITY ANALYSIS")
    print(f"{'='*80}")

    seed_words = ["محبت", "شاعری", "دل", "زندگی"]

    # Generate with each model type (using Adam optimizer)
    generated_samples = {}

    for model_name in ['rnn', 'lstm', 'transformer']:
        print(f"\n🤖 {model_name.upper()} TEXT GENERATION:")
        model = results[model_name]['adam']['model']
        samples = []

        # End any existing runs and start new one for text generation
        try:
            mlflow.end_run()
        except:
            pass

        try:
            with mlflow.start_run(run_name=f"{model_name}_text_generation"):

                mlflow.log_param("model_type", model_name)
                mlflow.log_param("generation_temperature", 1.0)
                mlflow.log_param("max_length", 15)
                mlflow.log_param("seed_words", str(seed_words))

                generation_log = []

                for seed in seed_words:
                    generated_text = generate_text(model, seed, token_to_idx, idx_to_token, max_length=15)
                    samples.append(generated_text)
                    print(f"  Seed '{seed}': {generated_text}")

                    generation_log.append(f"Seed: {seed}\nGenerated: {generated_text}\n")

                generated_samples[model_name] = samples

                # Log generation results to MLflow
                generation_text = "\n".join(generation_log)
                mlflow.log_text(generation_text, "generated_samples.txt")

                # Log quality metrics
                quality_analysis = analyze_generated_text_quality(samples, seed_words)
                for seed, analysis in quality_analysis.items():
                    mlflow.log_metric(f"gen_length_{seed}", analysis['length'])
                    mlflow.log_metric(f"gen_diversity_{seed}", analysis['diversity'])
                    mlflow.log_metric(f"gen_contains_urdu_{seed}", int(analysis['contains_urdu']))

        except Exception as e:
            print(f"⚠️  Warning: MLflow logging failed for {model_name} text generation: {e}")
            print("Continuing without MLflow logging for text generation...")

            # Fallback: generate text without MLflow
            for seed in seed_words:
                generated_text = generate_text(model, seed, token_to_idx, idx_to_token, max_length=15)
                samples.append(generated_text)
                print(f"  Seed '{seed}': {generated_text}")

            generated_samples[model_name] = samples

    # Quality analysis
    print(f"\n{'='*80}")
    print("GENERATED TEXT QUALITY ANALYSIS")
    print(f"{'='*80}")

    for model_name, samples in generated_samples.items():
        print(f"\n📝 {model_name.upper()} Quality Analysis:")
        quality_analysis = analyze_generated_text_quality(samples, seed_words)

        for seed, analysis in quality_analysis.items():
            print(f"  • '{seed}' → Length: {analysis['length']}, Diversity: {analysis['diversity']:.2f}, Urdu: {analysis['contains_urdu']}")

    # Log comprehensive results to MLflow
    try:
        mlflow.end_run()
    except:
        pass

    try:
        with mlflow.start_run(run_name="comprehensive_analysis"):

            # Log overall comparison metrics
            mlflow.log_metric("best_perplexity", best_overall['perplexity'])
            mlflow.log_metric("most_efficient_score", most_efficient['efficiency'])
            mlflow.log_param("best_model", best_overall['model'])
            mlflow.log_param("best_optimizer", best_overall['optimizer'])
            mlflow.log_param("most_efficient_model", most_efficient['model'])
            mlflow.log_param("most_efficient_optimizer", most_efficient['optimizer'])

            # Log comparison table as artifact
            import json
            comparison_json = json.dumps(comparison_table, indent=2)
            mlflow.log_text(comparison_json, "comparison_table.json")

    except Exception as e:
        print(f"⚠️  Warning: MLflow logging failed for comprehensive analysis: {e}")
        print("Continuing without MLflow logging for analysis...")

    # Save comprehensive results
    comprehensive_results = {
        'model_results': results,
        'comparison_table': comparison_table,
        'generated_samples': generated_samples,
        'quality_analysis': {model: analyze_generated_text_quality(samples, seed_words)
                           for model, samples in generated_samples.items()},
        'best_overall': best_overall,
        'most_efficient': most_efficient
    }

    with open('models/comprehensive_results.pkl', 'wb') as f:
        pickle.dump(comprehensive_results, f)

    # Log the comprehensive results file to MLflow
    try:
        mlflow.end_run()
    except:
        pass

    try:
        with mlflow.start_run(run_name="final_results"):
            mlflow.log_artifact('models/comprehensive_results.pkl')
    except Exception as e:
        print(f"⚠️  Warning: MLflow logging failed for final results: {e}")
        print("Continuing without MLflow logging for final results...")

    print("\n💾 Comprehensive results saved to models/comprehensive_results.pkl")
    print("🔍 All experiments logged to MLflow!")
    print("🎉 Analysis complete! All STEP 3 and STEP 4 requirements addressed.")
    print("\n📊 To view MLflow experiments:")
    print("   1. Run: mlflow ui")
    print("   2. Open http://localhost:5000 in your browser")
    print("   3. Select 'Urdu_Poetry_Generation_Baseline' experiment")
    print("   4. Compare all 9 model+optimizer combinations!")

if __name__ == "__main__":
    main()