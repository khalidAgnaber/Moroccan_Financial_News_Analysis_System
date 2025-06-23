"""
Custom Transformer model for sentiment analysis.
"""
import os
import sys
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, Dense, Dropout, LayerNormalization, 
    MultiHeadAttention, GlobalAveragePooling1D
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.base_model import BaseSentimentModel
from utils.logger import setup_logger

logger = setup_logger("transformer_model")

class TransformerBlock(tf.keras.layers.Layer):
    """Transformer block with multi-head attention and feed-forward network."""
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
    
    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class TokenAndPositionEmbedding(tf.keras.layers.Layer):
    """Embedding layer for tokens and positions."""
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = Embedding(input_dim=maxlen, output_dim=embed_dim)
    
    def call(self, inputs):
        maxlen = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        token_embeddings = self.token_emb(inputs)
        return token_embeddings + positions

class TransformerSentimentModel(BaseSentimentModel):
    def __init__(self, input_shape, vocab_size, embed_dim=128, num_heads=8, ff_dim=128, num_layers=4):
        super().__init__("transformer_sentiment", input_shape)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_layers = num_layers
    
    def build_model(self):
        """Build the Transformer model architecture."""
        # Input layer
        inputs = Input(shape=(self.input_shape,))
        
        # Token and position embedding
        embedding_layer = TokenAndPositionEmbedding(self.input_shape, self.vocab_size + 1, self.embed_dim)
        x = embedding_layer(inputs)
        
        # Transformer blocks
        for _ in range(self.num_layers):
            x = TransformerBlock(self.embed_dim, self.num_heads, self.ff_dim)(x)
        
        # Global pooling and output layers
        x = GlobalAveragePooling1D()(x)
        x = Dropout(0.1)(x)
        x = Dense(64, activation="relu")(x)
        x = Dropout(0.1)(x)
        outputs = Dense(3, activation="softmax")(x)  # 3 classes: negative, neutral, positive
        
        # Create model
        self.model = Model(inputs=inputs, outputs=outputs)
        
        logger.info("Transformer model built successfully")
        return self.model

if __name__ == "__main__":
    # Test model creation
    vocab_size = 10000
    input_length = 512
    
    # Create and build model
    model = TransformerSentimentModel(input_length, vocab_size)
    model.build_model()
    model.compile_model()
    
    # Print model summary
    model.model.summary()