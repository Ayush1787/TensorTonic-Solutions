
import numpy as np

class BertEmbeddings:
    """
    BERT Embeddings = Token + Position + Segment
    """
    
    def __init__(self, vocab_size: int, max_position: int, hidden_size: int):
        self.hidden_size = hidden_size
        
        # Token embeddings
        self.token_embeddings = np.random.randn(vocab_size, hidden_size) * 0.02
        
        # Position embeddings
        self.position_embeddings = np.random.randn(max_position, hidden_size) * 0.02
        
        # Segment embeddings
        self.segment_embeddings = np.random.randn(2, hidden_size) * 0.02
    
    def forward(self, token_ids: np.ndarray, segment_ids: np.ndarray) -> np.ndarray:
        
        batch_size, seq_len = token_ids.shape
        
        # Token embeddings
        tok_emb = self.token_embeddings[token_ids]
        
        # Position indices
        positions = np.arange(seq_len)
        pos_emb = self.position_embeddings[positions]
        
        # Expand position embeddings to batch
        pos_emb = pos_emb[np.newaxis, :, :]
        
        # Segment embeddings
        seg_emb = self.segment_embeddings[segment_ids]
        
        # Final BERT embedding
        embeddings = tok_emb + pos_emb + seg_emb
        
        return embeddings