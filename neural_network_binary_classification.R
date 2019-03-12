X <- as.matrix(t(iris[,1:4]))
Y <- as.numeric(iris$Species == "setosa")

sigmoid <- function(x) {
  return(1/(1+exp(-x)))
}
tan_h <- function(x) {
  return((exp(x) - exp(-x))/(exp(x) + exp(-x)))
}

layer_sizes <- function(X,Y) {
  n_x <- nrow(X)
  n_h <- 4 
  n_y <- 1
  return(list(n_x = n_x,
              n_h = n_h,
              n_y = n_y))
}

initialize_parameters <- function(n_x, n_h, n_y) {
  W1 <- matrix(runif(n_x*n_h) * 0.01, nrow = n_h, ncol = n_x)
  b1 <- rep(0, n_h)
  W2 <- matrix(runif(n_h*n_y) * 0.01, nrow = n_y, ncol = n_h)
  b2 <- rep(0, n_y)
  
  return(list(W1 = W1,
              b1 = b1,
              W2 = W2,
              b2 = b2))
}

forward_propagation <- function(X, parameters) {
  W1 <- parameters$W1
  b1 <- parameters$b1
  W2 <- parameters$W2
  b2 <- parameters$b2
  
  Z1 <- (W1 %*% X) + b1
  A1 <- tanh(Z1)
  Z2 <- (W2 %*% A1) + b2
  A2 <- sigmoid(Z2)
  
  cache <- list(Z1 = Z1,
              A1 = A1,
              Z2 = Z2,
              A2 = A2)
  return(cache)
}

compute_cost <- function(A2, Y) {
  m = length(Y)
 
  cost = (-1/m) * sum(
    (Y*log(A2) + ((1-Y)*log(1 - A2)))
     )
  
  return(cost)
}

backward_propagation <- function(parameters, cache, X, Y) {
  m = length(Y)
  
  W1 = parameters$W1
  b1 = parameters$b1
  W2 = parameters$W2
  b2 = parameters$b2
  
  A1 = cache$A1
  A2 = cache$A2
  
  dZ2 = A2 - Y
  dW2 = (1/m) * (dZ2 %*% t(A1))
  db2 = (1/m) * sum(dZ2)
  dZ1 = t(W2) %*% dZ2 * (1 - A1^2)
  dW1 = (1/m) * (dZ1 %*% t(X))
  db1 = (1/m) * sum(dZ1)
  
  grads = list(dW1 = dW1,
               db1 = db1,
               dW2 = dW2,
               db2 = db2)
  
  return(grads)
}

update_parameters <- function(parameters, grads, learning_rate = 0.1) {
  W1 = parameters$W1
  b1 = parameters$b1
  W2 = parameters$W2
  b2 = parameters$b2
  
  dW1 = grads$dW1
  db1 = grads$db1
  dW2 = grads$dW2
  db2 = grads$db2
  
  W1 = W1 - learning_rate * dW1
  b1 = b1 - learning_rate * db1
  W2 = W2 - learning_rate * dW2
  b2 = b2 - learning_rate * db2
  
  parameters = list(W1 = W1,
                    b1 = b1,
                    W2 = W2,
                    b2 = b2)
  return(parameters)
}

nn_model <- function(X, Y, n_h, num_iterations = 10000, print_cost = FALSE) {
  set.seed(3)
  layerSizes = layer_sizes(X, Y)
  n_x = layerSizes$n_x
  n_y = layerSizes$n_y
  
  parameters = initialize_parameters(n_x, n_h, n_y)
  W1 = parameters$W1
  b1 = parameters$b1
  W2 = parameters$W2
  b2 = parameters$b2
  
  for(i in 1:num_iterations) {
    cache = forward_propagation(X, parameters)
    A2 = cache$A2
    
    cost = compute_cost(A2, Y)
    
    grads = backward_propagation(parameters, cache, X, Y)
    
    parameters = update_parameters(parameters, grads)
    
   if(print_cost && i %% 1000 == 0) {print(paste0("Cost after iteration ", i, " is ", cost))}
    
  }
  
  return(parameters)
}

predict <- function(parameters, X) {
  cache <- forward_propagation(X, parameters)
  predictions <- as.numeric(cache$A2 > 0.5)
  return(predictions)
}


accuracy <- function(predictions, Y) {
  
  accuracy <- sum(Y * predictions + (1-Y)*(1 - predictions))/length(Y)
  return(accuracy)
}
#================================================#


my_model_params <- nn_model(X, Y, n_h = 4, num_iterations = 10000, print_cost = TRUE )
predictions <- predict(my_model_params, X)
accuracy(predictions, Y)


