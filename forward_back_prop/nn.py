import numpy as np
import matplotlib.pyplot as plt


from activation import sigmoid, relu
from cost import compute_cost_binary_classification
from derivatives import sigmoid_backward, relu_backward, linear_backward




def initialize_parameters_deep(layer_dims):
  """
  Arguments:
  layer_dims -- python array (list) containing the dimensions of each layer in our network INCLUDING the input layer. E.g. [256, 128, 64, 1]
  
  Returns:
  parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
    bl -- bias vector of shape (layer_dims[l], 1)
  """
  
  np.random.seed(3)
  parameters = {}
  L = len(layer_dims)            # number of layers in the network

  for l in range(1, L):
    nx = layer_dims[l]
    nx_prev = layer_dims[l - 1]
    parameters['W' + str(l)] = np.random.randn(nx, nx_prev) * 0.01
    parameters['b' + str(l)] = np.zeros((nx, 1))
    assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
    assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))
 
  return parameters



def linear_forward(A, W, b):
  """
  Implement the linear part of a layer's forward propagation.

  Arguments:
  A -- activations from previous layer (or input data): (size of previous layer, number of examples)
  W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
  b -- bias vector, numpy array of shape (size of the current layer, 1)

  Returns:
  Z -- the input of the activation function, also called pre-activation parameter 
  cache -- a python tuple containing "A", "W" and "b" ; stored for computing the backward pass efficiently
  """

  Z = np.dot(W, A) + b

  assert(Z.shape == (W.shape[0], A.shape[1]))
  cache = (A, W, b)
  
  return Z, cache



def linear_activation_forward(A_prev, W, b, activation):
  """
  Implement the forward propagation for the LINEAR->ACTIVATION layer

  Arguments:
  A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
  W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
  b -- bias vector, numpy array of shape (size of the current layer, 1)
  activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

  Returns:
  A -- the output of the activation function, also called the post-activation value 
  cache -- a python tuple containing "linear_cache" and "activation_cache";
            stored for computing the backward pass efficiently
  """

  Z, linear_cache = linear_forward(A_prev, W, b)

  if activation == 'sigmoid':
    A, activation_cache = sigmoid(Z)
  elif activation == 'relu':
    A, activation_cache = relu(Z)
  else:
    raise ValueError('Invalid activation')

  assert (A.shape == (W.shape[0], A_prev.shape[1]))
  cache = (linear_cache, activation_cache)

  return A, cache




def linear_activation_backward(dA, cache, activation):
  """
  Implement the backward propagation for the LINEAR->ACTIVATION layer.
  
  Arguments:
  dA -- post-activation gradient for current layer l 
  cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
  activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
  
  Returns:
  dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
  dW -- Gradient of the cost with respect to W (current layer l), same shape as W
  db -- Gradient of the cost with respect to b (current layer l), same shape as b
  """
  linear_cache, activation_cache = cache
  
  if activation == "relu":
    dZ = relu_backward(dA, activation_cache)
    dA_prev, dW, db  = linear_backward(dZ, linear_cache)
      
  elif activation == "sigmoid":
    dZ = sigmoid_backward(dA, activation_cache)
    dA_prev, dW, db  = linear_backward(dZ, linear_cache)
  else:
    raise ValueError('Invalid activation requested')
  
  return dA_prev, dW, db



def L_model_forward(X, parameters):
  """
  Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
  
  Arguments:
  X -- data, numpy array of shape (input size, number of examples)
  parameters -- output of initialize_parameters_deep()
  
  Returns:
  AL -- last post-activation value
  caches -- list of caches containing:
    every cache of linear_activation_forward() (there are L-1 of them, indexed from 0 to L-1)
  """

  caches = []
  A = X
  L = len(parameters) // 2                  # number of layers in the neural network
  
  # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
  for l in range(1, L):
      A_prev = A 
      A, cache = linear_activation_forward(A, parameters['W' + str(l)], parameters['b' + str(l)], 'relu')
      caches.append(cache)
  
  # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
  AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], 'sigmoid')
  caches.append(cache)
  
  assert(AL.shape == (1,X.shape[1]))
          
  return AL, caches



def L_model_backward(AL, Y, caches):
  """
  Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
  
  Arguments:
  AL -- probability vector, output of the forward propagation (L_model_forward())
  Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
  caches -- list of caches containing:
    every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
    the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
  
  Returns:
  grads -- A dictionary with the gradients
  grads["dA" + str(l)] = ... 
  grads["dW" + str(l)] = ...
  grads["db" + str(l)] = ... 
  """
  grads = {}
  L = len(caches) # the number of layers
  m = AL.shape[1]
  Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
  
  # Initializing the backpropagation
  dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
  
  # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "dAL, current_cache". Outputs: "grads["dAL-1"], grads["dWL"], grads["dbL"]
  current_cache = caches[L - 1]
  grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(
    dAL,
    current_cache,
    activation='sigmoid'
  )

  # Loop from l=L-2 to l=0
  for l in reversed(range(L-1)):
    current_cache = caches[l]
    # lth layer: (RELU -> LINEAR) gradients.
    # Inputs: "grads["dA" + str(l + 1)], current_cache". Outputs: "grads["dA" + str(l)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)] 
    grads["dA" + str(l)], grads["dW" + str(l + 1)], grads["db" + str(l + 1)] = linear_activation_backward(
      grads['dA' + str(l + 1)],
      current_cache,
      activation='relu'
    )

  return grads



def update_parameters(parameters, grads, learning_rate):
  """
  Update parameters using gradient descent
  
  Arguments:
  parameters -- python dictionary containing your parameters 
  grads -- python dictionary containing your gradients, output of L_model_backward
  
  Returns:
  parameters -- python dictionary containing your updated parameters 
  parameters["W" + str(l)] = ... 
  parameters["b" + str(l)] = ...
  """
  
  L = len(parameters) // 2 # number of layers in the neural network

  # Update rule for each parameter. Use a for loop.
  for l in range(L):
    parameters['W' + str(l + 1)] = parameters['W' + str(l + 1)] - learning_rate * grads['dW' + str(l + 1)]
    parameters['b' + str(l + 1)] = parameters['b' + str(l + 1)] - learning_rate * grads['db' + str(l + 1)]

  return parameters



def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):#lr was 0.009
  """
  Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
  
  Arguments:
  X -- data, numpy array of shape (num_px * num_px * 3, number of examples)
  Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
  layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
  learning_rate -- learning rate of the gradient descent update rule
  num_iterations -- number of iterations of the optimization loop
  print_cost -- if True, it prints the cost every 100 steps
  
  Returns:
  parameters -- parameters learnt by the model. They can then be used to predict.
  """

  np.random.seed(1)
  costs = []
  
  # Parameters initialization. (≈ 1 line of code)
  parameters = initialize_parameters_deep(layers_dims)
  
  # Loop (gradient descent)
  for i in range(0, num_iterations):
    # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
    AL, caches = L_model_forward(X, parameters)
    
    # Compute cost.
    cost = compute_cost_binary_classification(AL, Y)

    # Backward propagation.
    grads = L_model_backward(AL, Y, caches)

    # Update parameters.
    parameters = update_parameters(
      parameters,
      grads,
      learning_rate
    )
            
    # Print the cost every 100 training example
    if print_cost and i % 100 == 0:
      print ("Cost after iteration %i: %f" %(i, cost))
    if print_cost and i % 100 == 0:
      costs.append(cost)
          
  # plot the cost
  plt.plot(np.squeeze(costs))
  plt.ylabel('cost')
  plt.xlabel('iterations (per hundreds)')
  plt.title("Learning rate =" + str(learning_rate))
  plt.show()
  
  return parameters
