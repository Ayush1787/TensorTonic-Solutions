import torch
import torch.nn as nn
import math

def create_embedding_layer(vocab_size: int, d_model: int) -> nn.Embedding:
    """
    Create an embedding layer.
    """
    embedding = nn.Embedding(vocab_size, d_model)

    # initialize weights (scaled initialization)
    nn.init.normal_(embedding.weight, mean=0, std=1/math.sqrt(d_model))

    return embedding


def embed_tokens(embedding: nn.Embedding, tokens: torch.Tensor, d_model: int) -> torch.Tensor:
    """
    Convert token indices to scaled embeddings.
    """
    # lookup embeddings
    x = embedding(tokens)

    # scale embeddings
    x = x * math.sqrt(d_model)

    return x

vocab_size = 100
d_model = 64

embedding = create_embedding_layer(vocab_size, d_model)

tokens = torch.tensor([5,12,3])

output = embed_tokens(embedding, tokens, d_model)

print(output.shape)