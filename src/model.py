# model.py
import numpy as np
from src.activation import activations
from src.loss import loss_functions

class NeuralNetwork:
    def __init__(self, 
                 input_size, 
                 hidden_size, 
                 num_layers, 
                 output_size, 
                 activation="ReLU", 
                 weight_init="Xavier", 
                 loss_type="cross_entropy"):
        """
        Args:
          input_size   : Number of input features
          hidden_size  : Number of neurons in each hidden layer
          num_layers   : Number of hidden layers
          output_size  : Number of output neurons
          activation   : Name of activation function (e.g. "ReLU")
          weight_init  : Weight initialization scheme ("random" or "Xavier")
          loss_type    : Loss function name ("cross_entropy" or "mean_squared_error")
        """

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        # Load activation and its derivative from activations.py
        if activation not in activations:
            raise ValueError(f"Unsupported activation function '{activation}'")
        self.activation, self.activation_deriv = activations[activation]

        # Load loss and its derivative from losses.py
        if loss_type not in loss_functions:
            raise ValueError(f"Unsupported loss function '{loss_type}'")
        self.loss_func, self.loss_deriv = loss_functions[loss_type]
        self.loss_type = loss_type

        # Initialize parameters (weights and biases)
        self.parameters = {}
        layer_sizes = [input_size] + [hidden_size]*num_layers + [output_size]

        for i in range(1, len(layer_sizes)):
            if weight_init == "Xavier":
                limit = np.sqrt(6 / (layer_sizes[i - 1] + layer_sizes[i]))
                self.parameters[f"W{i}"] = np.random.uniform(-limit, limit, 
                                         (layer_sizes[i - 1], layer_sizes[i]))
            else:
                # Simple random init with small variance
                self.parameters[f"W{i}"] = np.random.randn(layer_sizes[i - 1], 
                                                           layer_sizes[i]) * 0.01

            # Biases init to zero
            self.parameters[f"b{i}"] = np.zeros((1, layer_sizes[i]))

    def forward_pass(self, X):
        """
        Forward pass  
        Returns:
          a_list: List of pre-activation values [a1, a2, ..., aL].
          h_list: List of post-activation values [h0, h1, ..., hL].
          y_hat : Final output (softmax or linear), depending on loss.
        """
        a_list = []
        h_list = []

        # h0 = input X
        h_list.append(X)

        # 1) Forward through hidden layers
        for i in range(1, self.num_layers + 1):
            a_k = np.dot(h_list[-1], self.parameters[f"W{i}"]) + self.parameters[f"b{i}"]
            a_list.append(a_k)
            h_k = self.activation(a_k)
            h_list.append(h_k)

        # 2) Output layer (layer L = num_layers + 1)
        a_L = np.dot(h_list[-1], self.parameters[f"W{self.num_layers + 1}"]) + \
              self.parameters[f"b{self.num_layers + 1}"]
        a_list.append(a_L)

        # For cross_entropy, apply softmax
        if self.loss_type == "cross_entropy":
            # numerical stability trick
            exp_scores = np.exp(a_L - np.max(a_L, axis=1, keepdims=True))
            y_hat = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        else:
            # For mean_squared_error, treat output as linear (identity)
            y_hat = a_L

        h_list.append(y_hat)
        return a_list, h_list, y_hat
    
    def forward(self, X):
        """
        Only Final predictions.
        """
        _, _, y_hat = self.forward_pass(X)
        return y_hat

    def compute_loss(self, y_pred, y_true):
        """
        Compute the loss given predictions and true labels.
        """
        return self.loss_func(y_pred, y_true)

    def backward_pass(self, a_list, h_list, Y, y_hat):
        """
        Backward pass using the lists of pre-activations (a_list) and 
        activations (h_list) from forward_pass().
        Returns:
          grads: Dictionary of gradients dW1, db1, ..., dWL, dbL.
        """
        grads = {}
        m = Y.shape[0]
        L = self.num_layers + 1  # total layers = hidden_layers + 1

        # 1) Compute dZ for output layer
        #    cross_entropy_deriv or mean_squared_error_deriv
        dZ = self.loss_deriv(y_hat, Y)  

        # Grad for final layer W_L, b_L
        # h_list[-2] is h_{L-1}
        grads[f"dW{L}"] = np.dot(h_list[-2].T, dZ) / m
        grads[f"db{L}"] = np.sum(dZ, axis=0, keepdims=True) / m

        # 2) Backprop through hidden layers in reverse
        for k in range(L - 1, 0, -1):
            # derivative wrt h_{k-1}
            dA = np.dot(dZ, self.parameters[f"W{k+1}"].T)

            # derivative wrt a_{k-1} = dA * g'(a_{k-1})
            a_k_minus_1 = a_list[k - 1]  
            dZ = dA * self.activation_deriv(a_k_minus_1)

            # Grad for W_k, b_k
            grads[f"dW{k}"] = np.dot(h_list[k - 1].T, dZ) / m
            grads[f"db{k}"] = np.sum(dZ, axis=0, keepdims=True) / m

        return grads
    
