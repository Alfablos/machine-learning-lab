import numpy as np


def zero_pad(X, pad):
    """
    Pad with zeros all images of the dataset X. The padding is applied to the height and width of an image,
    as illustrated in Figure 1.

    Argument:
    X -- python numpy array of shape (m, n_H, n_W, n_C) representing a batch of m images
    pad -- integer, amount of padding around each image on vertical and horizontal dimensions

    Returns:
    X_pad -- padded image of shape (m, n_H + 2 * pad, n_W + 2 * pad, n_C)
    """

    # (≈ 1 line)
    # X_pad = None
    # YOUR CODE STARTS HERE
    X_pad = np.pad(
        X,
        ((0, 0), (pad, pad), (pad, pad), (0, 0)),
        mode="constant",
        constant_values=(0, 0),
    )

    # YOUR CODE ENDS HERE

    return X_pad


def conv_single_step(a_slice_prev, W, b):
    """
    Apply one filter defined by parameters W on a single slice (a_slice_prev) of the output activation
    of the previous layer.

    Arguments:
    a_slice_prev -- slice of input data of shape (f, f, n_C_prev)
    W -- Weight parameters contained in a window - matrix of shape (f, f, n_C_prev)
    b -- Bias parameters contained in a window - matrix of shape (1, 1, 1)

    Returns:
    Z -- a scalar value, the result of convolving the sliding window (W, b) on a slice x of the input data
    """

    # Element-wise product between a_slice_prev and W. Do not add the bias yet.
    s = np.multiply(
        a_slice_prev, W
    )  # a_slice_prev and W HAVE THE SAME SIZE, no shifting needed
    # Sum over all entries of the volume s.
    Z = np.sum(s)
    # Add bias b to Z. Cast b to a float() so that Z results in a scalar value.
    Z = Z + float(b)

    return Z


def conv_forward(A_prev, W, b, hparameters, activation):
    """
    Implements the forward propagation for a convolution function

    Arguments:
    A_prev -- output activations of the previous layer,
        numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    W -- Weights, numpy array of shape (f, f, n_C_prev, n_C)
    b -- Biases, numpy array of shape (1, 1, 1, n_C)
    hparameters -- python dictionary containing "stride" and "pad"

    Returns:
    Z -- conv output, numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache of values needed for the conv_backward() function
    """

    # Retrieve dimensions from A_prev's shape (≈1 line)
    # (m, n_H_prev, n_W_prev, n_C_prev) = None

    # Retrieve dimensions from W's shape (≈1 line)
    # (f, f, n_C_prev, n_C) = None
    # Retrieve information from "hparameters" (≈2 lines)
    # stride = None
    # pad = None

    # Compute the dimensions of the CONV output volume using the formula given above.
    # Hint: use int() to apply the 'floor' operation. (≈2 lines)
    # n_H = None
    # n_W = None

    # Initialize the output volume Z with zeros. (≈1 line)
    # Z = None

    # Create A_prev_pad by padding A_prev
    # A_prev_pad = None

    # for i in range(None):               # loop over the batch of training examples
    # a_prev_pad = None               # Select ith training example's padded activation
    # for h in range(None):           # loop over vertical axis of the output volume
    # Find the vertical start and end of the current "slice" (≈2 lines)
    # vert_start = None
    # vert_end = None

    # for w in range(None):       # loop over horizontal axis of the output volume
    # Find the horizontal start and end of the current "slice" (≈2 lines)
    # horiz_start = None
    # horiz_end = None

    # for c in range(None):   # loop over channels (= #filters) of the output volume

    # Use the corners to define the (3D) slice of a_prev_pad (See Hint above the cell). (≈1 line)
    # a_slice_prev = None

    # Convolve the (3D) slice with the correct filter W and bias b, to get back one output neuron. (≈3 line)
    # weights = None
    # biases = None
    # Z[i, h, w, c] = None

    (m, n_H_prev, n_W_prev, _n_C_prev) = (
        A_prev.shape[0],
        A_prev.shape[1],
        A_prev.shape[2],
        A_prev.shape[3],
    )

    (f, f, _n_C_prev, n_C) = (W.shape[0], W.shape[1], W.shape[2], W.shape[3])

    stride = hparameters["stride"]

    pad = hparameters["pad"]

    n_H = int((n_H_prev + 2 * pad - f) / stride + 1)

    n_W = int((n_W_prev + 2 * pad - f) / stride + 1)

    Z = np.zeros(shape=(m, n_H, n_W, n_C))

    A_prev_pad = zero_pad(A_prev, pad)

    for i in range(m):
        a_prev_pad = A_prev_pad[
            i
        ]  # Select ith training example's padded activation
        for h in range(n_H):  # loop over vertical axis of the output volume
            vert_start = h * stride
            vert_end = vert_start + f

            for w in range(n_W):  # loop over horizontal axis of the output volume
                horiz_start = (
                    w * stride
                )  # Find the horizontal start and end of the current "slice"
                horiz_end = horiz_start + f

                for c in range(
                    n_C
                ):  # loop over channels (= #filters) of the output volume
                    # Use the corners to define the (3D) slice of a_prev_pad for example i
                    # A_prev_pad is 4D because it contains 3D matrices for m examples
                    a_slice_prev = A_prev_pad[
                        vert_start:vert_end, horiz_start:horiz_end, :
                    ]  # use the padded version so we don't loose information!

                    weights = W[:, :, :, c]
                    biases = b[:, :, :, c]
                    Z[i, h, w, c] = conv_single_step(a_slice_prev, weights, biases)

    assert Z.shape == (m, n_H, n_W, n_C)

    # Save information in "cache" for the backprop
    cache = (A_prev, W, b, hparameters)

    raise RuntimeError('Add activation step here!')

    return Z, cache



def pool_forward(A_prev, hparameters, mode = "max"):
    """
    Implements the forward pass of the pooling layer
    
    Arguments:
    A_prev -- Input data, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    hparameters -- python dictionary containing "f" and "stride"
    mode -- the pooling mode you would like to use, defined as a string ("max" or "average")
    
    Returns:
    A -- output of the pool layer, a numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache used in the backward pass of the pooling layer, contains the input and hparameters 
    """
    
    # Retrieve dimensions from the input shape
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    
    # Retrieve hyperparameters from "hparameters"
    f = hparameters["f"]
    stride = hparameters["stride"]
    
    # Define the dimensions of the output
    n_H = int(1 + (n_H_prev - f) / stride)
    n_W = int(1 + (n_W_prev - f) / stride)
    n_C = n_C_prev                          # Pooling doesn't change the output!
    
    # Initialize output matrix A
    A = np.zeros((m, n_H, n_W, n_C))              
    
    # for i in range(None):                         # loop over the training examples
        # for h in range(None):                     # loop on the vertical axis of the output volume
            # Find the vertical start and end of the current "slice" (≈2 lines)
            # vert_start = None
            # vert_end = None
            
            # for w in range(None):                 # loop on the horizontal axis of the output volume
                # Find the vertical start and end of the current "slice" (≈2 lines)
                # horiz_start = None
                # horiz_end = None
                
                # for c in range (None):            # loop over the channels of the output volume
                    
                    # Use the corners to define the current slice on the ith training example of A_prev, channel c. (≈1 line)
                    # a_prev_slice = None
                    
                    # Compute the pooling operation on the slice. 
                    # Use an if statement to differentiate the modes. 
                    # Use np.max and np.mean.
                    # if mode == "max":
                        # A[i, h, w, c] = None
                    # elif mode == "average":
                        # A[i, h, w, c] = None
    
    for i in range(m):
        for h in range(n_H):
            vert_start = i * stride
            vert_end = vert_start + f

            for w in range(n_W):
                horiz_start = w * stride
                horiz_end = horiz_start + f

                for c in range(n_C):
                    # for example i and channel c: slice the resulting 2D array; No padding needed since output dimensions are not less
                    a_prev_slice = A_prev[i, vert_start:vert_end, horiz_start:horiz_end, c]
                    if mode == 'max':
                        A[i, h, w, c] = np.max(a_prev_slice)
                    elif mode == 'average':
                        A[i, h, w, c] = np.mean(a_prev_slice)
                    else:
                        raise ValueError('The only allowed pooling modes are \'max\' and \'average\'.')
    
    # YOUR CODE ENDS HERE
    
    # Store the input and hparameters in "cache" for pool_backward()
    cache = (A_prev, hparameters)
    
    # Making sure your output shape is correct
    #assert(A.shape == (m, n_H, n_W, n_C))
    
    return A, cache