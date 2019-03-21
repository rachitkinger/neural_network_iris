# will use College dataset from ISLR to predict whether a college is public or private
College <- ISLR::College
# create X (remove 'Name' variable) and transpose
X <- t(College[,-1])
# create Y as a numeric vector 1 = Private, 0 = Not Private
Y <- as.numeric(College$Private == "Yes")

# architecture 
# L layer deep neural network
# L-1 ReLU activations and 1 sigmoid activation   

sigmoid <- function(x) {
  return(1/(1+exp(-x)))
}
tan_h <- function(x) {
  return((exp(x) - exp(-x))/(exp(x) + exp(-x)))
}
relu <- function(Z) {
  A = max(0, Z)
  return(A)
}

# layer_dims = vector of length L+1 (includes i/p layer or layer 0)
# each element of layer_dims represents the num of units in each layer
initialize_parameters_deep <- function(layer_dims) {
  set.seed(3)
  parameters <- list()
  L = length(layer_dims)
  
  for(l in 2:L) {
    parameters[[paste0("W",l-1)]] <- matrix(runif(layer_dims[l]*layer_dims[l-1]) * 0.01, nrow = layer_dims[l], ncol = layer_dims[l-1])
    parameters[[paste0("b",l-1)]] <- rep(0, layer_dims[l])
  }
  
  return(parameters)
}

# forward prop==========================================
# 3 functions
# 1. linear forward i.e. z = w.x + b
# 2. linear activation i.e. a = g(z)
# 3. deep model that loops through L layers  
linear_forward <- function(A, W, b) {
  # Arguments:
  #   A -- activations from previous layer (or input data): (size of previous layer, number of examples)
  # W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
  # b -- bias vector, numpy array of shape (size of the current layer, 1)
  Z = W %*% A + b
  cache = list(Z = Z,
               A = A,
               W = W,
               b = b)
  return(cache)
}

linear_activation_forward <- function(A_prev, W, b, activation) {
  if(activation == "sigmoid") {
    linear_cache <- linear_forward(A_prev, W , b)
    A <- sigmoid(linear_cache$Z)
    activation_cache <- list(A = A,
                             Z = linear_cache$Z)
  }
  
  if(activation == "relu"){
    linear_cache = linear_forward(A_prev, W, b)
    A = relu(linear_cache$Z)
    activation_cache <- list(A = A,
                             Z = linear_cache$Z)
  }
  
  cache = list(A = A,
               linear_cache = linear_cache,
               activation_cache = activation_cache)
  
  return(cache)
}

L_model_forward <- function(X, parameters) {
  caches = list()
  A = X
  L = length(parameters) / 2
  
}
