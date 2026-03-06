import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """

    x = np.array(x, dtype=float)

    if rng is None:
        rand = np.random.random(x.shape)
    else:
        rand = rng.random(x.shape)

    keep_prob = 1 - p

    mask = rand < keep_prob

    scale = 1.0 / keep_prob

    dropout_pattern = mask.astype(float) * scale

    output = x * dropout_pattern

    return output, dropout_pattern