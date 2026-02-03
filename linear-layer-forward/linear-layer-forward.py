def linear_layer_forward(X, W, b):
    """
    Compute the forward pass of a linear (fully connected) layer.
    Y = XW + b
    """
    n = len(X)          # number of samples
    d_out = len(W[0])   # output dimension
    d_in = len(W)       # input dimension

    # Initialize output with zeros
    Y = [[0 for _ in range(d_out)] for _ in range(n)]

    # Compute Y = XW + b
    for i in range(n):              # over samples
        for j in range(d_out):      # over output dimensions
            for k in range(d_in):   # over input dimensions
                Y[i][j] += X[i][k] * W[k][j]
            Y[i][j] += b[j]         # add bias

    return Y
