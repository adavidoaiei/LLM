import tensorflow as tf
import numpy as np
import re
import os
from typing import Dict, List, Tuple, Optional

class MultiHeadAttention(tf.keras.layers.Layer):
    """Multi-head self-attention mechanism."""
    
    def __init__(self, d_model: int, num_heads: int, dropout_rate: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        
        assert d_model % num_heads == 0
        self.depth = d_model // num_heads
        
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        
    def split_heads(self, x: tf.Tensor, batch_size: int) -> tf.Tensor:
        """Split the last dimension into (num_heads, depth)."""
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def scaled_dot_product_attention(self, q: tf.Tensor, k: tf.Tensor, v: tf.Tensor, mask: Optional[tf.Tensor] = None) -> Tuple[tf.Tensor, tf.Tensor]:
        """Calculate the attention weights."""
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        
        # Scale matmul_qk
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        
        # Add the mask to the scaled tensor
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        
        # Softmax is normalized on the last axis
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = tf.matmul(attention_weights, v)
        return output, attention_weights
    
    def call(self, v: tf.Tensor, k: tf.Tensor, q: tf.Tensor, mask: Optional[tf.Tensor] = None, training: bool = False) -> tf.Tensor:
        batch_size = tf.shape(q)[0]
        
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v, mask)
        
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        
        output = self.dense(concat_attention)
        return output

class FeedForward(tf.keras.layers.Layer):
    """Position-wise feed-forward network."""
    
    def __init__(self, d_model: int, dff: int, dropout_rate: float = 0.1):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(dff, activation='relu')
        self.dense2 = tf.keras.layers.Dense(d_model)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        
    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        return x

class EncoderLayer(tf.keras.layers.Layer):
    """Single encoder layer."""
    
    def __init__(self, d_model: int, num_heads: int, dff: int, dropout_rate: float = 0.1):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads, dropout_rate)
        self.ffn = FeedForward(d_model, dff, dropout_rate)
        
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        
    def call(self, x: tf.Tensor, mask: Optional[tf.Tensor] = None, training: bool = False) -> tf.Tensor:
        attn_output = self.mha(x, x, x, mask, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        
        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2

class PositionalEncoding(tf.keras.layers.Layer):
    """Positional encoding for transformer."""
    
    def __init__(self, position: int, d_model: int):
        super().__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)
        
    def get_angles(self, pos: np.ndarray, i: np.ndarray, d_model: int) -> np.ndarray:
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        return pos * angle_rates
    
    def positional_encoding(self, position: int, d_model: int) -> tf.Tensor:
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis],
                                   np.arange(d_model)[np.newaxis, :],
                                   d_model)
        
        # Apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        
        # Apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        
        pos_encoding = angle_rads[np.newaxis, ...]
        
        return tf.cast(pos_encoding, dtype=tf.float32)
    
    def call(self, x: tf.Tensor) -> tf.Tensor:
        return x + self.pos_encoding[:, :tf.shape(x)[1], :]

class TransformerLLM(tf.keras.Model):
    """Transformer-based Language Model."""
    
    def __init__(self, vocab_size: int, d_model: int = 512, num_heads: int = 8, 
                 num_layers: int = 6, dff: int = 2048, max_position: int = 10000,
                 dropout_rate: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(max_position, d_model)
        
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, dropout_rate)
                          for _ in range(num_layers)]
        
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.final_layer = tf.keras.layers.Dense(vocab_size)
        
    def create_look_ahead_mask(self, size: int) -> tf.Tensor:
        """Create look-ahead mask for decoder."""
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask
    
    def call(self, x: tf.Tensor, training: bool = False, mask: Optional[tf.Tensor] = None) -> tf.Tensor:
        seq_len = tf.shape(x)[1]
        
        # Create look-ahead mask
        if mask is None:
            mask = self.create_look_ahead_mask(seq_len)
        
        # Embedding and positional encoding
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = self.pos_encoding(x)
        
        x = self.dropout(x, training=training)
        
        # Pass through encoder layers
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, mask, training=training)
        
        # Final linear layer
        output = self.final_layer(x)
        return output

class TextTokenizer:
    """Simple text tokenizer for the LLM."""
    
    def __init__(self):
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.vocab_size = 0
        
    def fit(self, texts: List[str]) -> None:
        """Build vocabulary from texts."""
        # Simple tokenization
        words = set()
        for text in texts:
            # Clean and tokenize
            clean_text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
            words.update(clean_text.split())
        
        # Add special tokens
        special_tokens = ['<PAD>', '<START>', '<END>', '<UNK>']
        words = special_tokens + list(words)
        
        self.word_to_idx = {word: idx for idx, word in enumerate(words)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        self.vocab_size = len(words)
    
    def encode(self, text: str) -> List[int]:
        """Convert text to token indices."""
        clean_text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
        tokens = clean_text.split()
        return [self.word_to_idx.get(token, self.word_to_idx['<UNK>']) for token in tokens]
    
    def decode(self, indices: List[int]) -> str:
        """Convert token indices to text."""
        return ' '.join([self.idx_to_word.get(idx, '<UNK>') for idx in indices])

class LLMTrainer:
    """Trainer class for the LLM."""
    
    def __init__(self, model: TransformerLLM, tokenizer: TextTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
        # Loss and optimizer
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none')
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        
        # Metrics
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    
    def loss_function(self, real: tf.Tensor, pred: tf.Tensor) -> tf.Tensor:
        """Calculate loss with masking."""
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = self.loss_object(real, pred)
        
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        
        return tf.reduce_sum(loss_)/tf.reduce_sum(mask)
    
    def create_dataset(self, texts: List[str], seq_length: int = 128, batch_size: int = 32) -> tf.data.Dataset:
        """Create training dataset."""
        sequences = []
        
        for text in texts:
            tokens = self.tokenizer.encode(text)
            # Create sequences
            for i in range(len(tokens) - seq_length):
                sequences.append(tokens[i:i+seq_length+1])
        
        # Convert to numpy array
        sequences = np.array(sequences)
        
        # Input and target
        inputs = sequences[:, :-1]
        targets = sequences[:, 1:]
        
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices((inputs, targets))
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    @tf.function
    def train_step(self, inp: tf.Tensor, tar: tf.Tensor) -> None:
        """Single training step."""
        with tf.GradientTape() as tape:
            predictions = self.model(inp, training=True)
            loss = self.loss_function(tar, predictions)
        
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        self.train_loss(loss)
        self.train_accuracy(tar, predictions)
    
    def train(self, texts: List[str], epochs: int = 10, seq_length: int = 128, batch_size: int = 32) -> None:
        """Train the model."""
        dataset = self.create_dataset(texts, seq_length, batch_size)
        
        for epoch in range(epochs):
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()
            
            for batch, (inp, tar) in enumerate(dataset):
                self.train_step(inp, tar)
                
                if batch % 50 == 0:
                    print(f'Epoch {epoch + 1} Batch {batch} Loss {self.train_loss.result():.4f} Accuracy {self.train_accuracy.result():.4f}')
            
            print(f'Epoch {epoch + 1} Loss {self.train_loss.result():.4f} Accuracy {self.train_accuracy.result():.4f}')

class LLMGenerator:
    """Text generator using the trained LLM."""
    
    def __init__(self, model: TransformerLLM, tokenizer: TextTokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def generate(self, prompt: str, max_length: int = 100, temperature: float = 1.0) -> str:
        """Generate text based on prompt."""
        # Encode prompt
        tokens = self.tokenizer.encode(prompt)
        tokens = tf.expand_dims(tokens, 0)
        
        generated_tokens = []
        
        for _ in range(max_length):
            predictions = self.model(tokens, training=False)
            
            # Get the last token's predictions
            predictions = predictions[:, -1, :] / temperature
            
            # Sample from the distribution
            predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
            
            # Add to generated tokens
            generated_tokens.append(predicted_id)
            
            # Update tokens
            tokens = tf.concat([tokens, tf.expand_dims([predicted_id], 0)], axis=-1)
            
            # Stop if we generate end token
            if predicted_id == self.tokenizer.word_to_idx.get('<END>', -1):
                break
        
        # Decode the generated tokens
        return self.tokenizer.decode(generated_tokens)

# Example usage
def main():
    # Sample training data
    training_texts = [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is a subset of artificial intelligence",
        "Deep learning uses neural networks with multiple layers",
        "Natural language processing helps computers understand human language",
        "Transformers are a type of neural network architecture",
        "Python is a popular programming language for AI",
        "TensorFlow is an open source machine learning framework",
        "Data is the fuel that powers machine learning algorithms",
        "Artificial intelligence will transform many industries",
        "Computer vision enables machines to see and interpret images"
    ]
    
    print("Creating tokenizer...")
    tokenizer = TextTokenizer()
    tokenizer.fit(training_texts)
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    print("Creating model...")
    model = TransformerLLM(
        vocab_size=tokenizer.vocab_size,
        d_model=256,
        num_heads=8,
        num_layers=4,
        dff=1024,
        max_position=1000,
        dropout_rate=0.1
    )
    
    print("Creating trainer...")
    trainer = LLMTrainer(model, tokenizer)
    
    print("Training model...")
    trainer.train(training_texts, epochs=5, seq_length=32, batch_size=8)
    
    print("Creating generator...")
    generator = LLMGenerator(model, tokenizer)
    
    print("Generating text...")
    prompt = "machine learning"
    generated_text = generator.generate(prompt, max_length=20, temperature=0.8)
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated_text}")
    
    # Save model
    model.save_weights('llm_weights')
    print("Model saved!")

if __name__ == "__main__":
    main()
