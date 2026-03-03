import numpy as np

def rnn_cell(x_t: np.ndarray, h_prev: np.ndarray, 
             W_xh: np.ndarray, W_hh: np.ndarray, b_h: np.ndarray) -> np.ndarray:
    """
    Single RNN cell forward pass.

    h_t = tanh(W_hh h_prev + W_xh x_t + b_h)

    Shapes:
    x_t     : (batch_size, input_size)
    h_prev  : (batch_size, hidden_size)
    W_xh    : (hidden_size, input_size)
    W_hh    : (hidden_size, hidden_size)
    b_h     : (hidden_size,)
    
    Returns:
    h_t     : (batch_size, hidden_size)
    """

    # Linear transformation of previous hidden state
    hidden_part = h_prev @ W_hh.T

    # Linear transformation of current input
    input_part = x_t @ W_xh.T

    # Add bias (broadcasting over batch)
    h_t = np.tanh(hidden_part + input_part + b_h)

    return h_t