# optimizers.py
import numpy as np
import inspect

class SGD:
    def __init__(self, parameters, learning_rate=1e-3, weight_decay=0.0):
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        # No additional state needed for plain SGD

    def update(self, parameters, grads):
        for key in parameters.keys():
            grad = grads.get("d" + key, 0)
            parameters[key] -= self.learning_rate * (grad + self.weight_decay * parameters[key])

class Momentum:
    def __init__(self, parameters, learning_rate=1e-3, momentum=0.9, weight_decay=0.0):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        # Initialize velocity for each parameter to zeros
        self.v = {key: np.zeros_like(val) for key, val in parameters.items()}

    def update(self, parameters, grads):
        for key in parameters.keys():
            grad = grads.get("d" + key, 0)
            # Update velocity: v = beta * v + lr * (grad + weight_decay * param)
            self.v[key] = self.momentum * self.v[key] + self.learning_rate * (grad + self.weight_decay * parameters[key])
            # Update parameter: param = param - velocity
            parameters[key] -= self.v[key]

class NAG:
    def __init__(self, parameters, learning_rate=1e-3, momentum=0.9, weight_decay=0.0):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        # Initialize velocity for each parameter to zeros
        self.v = {key: np.zeros_like(val) for key, val in parameters.items()}

    def update(self, parameters, grads):
        for key in parameters.keys():
            grad = grads.get("d" + key, 0)
            # Store the current velocity
            v_prev = self.v[key].copy()
            # Update velocity using gradients computed at the "look-ahead" parameters
            self.v[key] = self.momentum * self.v[key] + self.learning_rate * (grad + self.weight_decay * parameters[key])
            # Update parameter using Nesterov’s accelerated gradient
            parameters[key] -= self.momentum * v_prev + (1 + self.momentum) * self.v[key]

class RMSprop:
    def __init__(self, parameters, learning_rate=1e-3, beta=0.5, epsilon=1e-6, weight_decay=0.0):
        self.learning_rate = learning_rate
        self.beta = beta
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        # Initialize squared gradient accumulators to zeros
        self.s = {key: np.zeros_like(val) for key, val in parameters.items()}

    def update(self, parameters, grads):
        for key in parameters.keys():
            grad = grads.get("d" + key, 0)
            # Update the running average of squared gradients
            self.s[key] = self.beta * self.s[key] + (1 - self.beta) * (grad ** 2)
            # Parameter update with weight decay
            parameters[key] -= self.learning_rate * (grad + self.weight_decay * parameters[key]) / (np.sqrt(self.s[key]) + self.epsilon)


class Adam:
    def __init__(self, parameters, learning_rate=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-6, weight_decay=0.0):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.t = 0
        # Initialize first and second moment estimates for each parameter
        self.m = {key: np.zeros_like(val) for key, val in parameters.items()}
        self.v = {key: np.zeros_like(val) for key, val in parameters.items()}

    def update(self, parameters, grads):
        self.t += 1
        for key in parameters.keys():
            grad = grads.get("d" + key, 0)
            # Update biased first moment estimate
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grad
            # Update biased second raw moment estimate
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grad ** 2)
            # Compute bias-corrected first moment estimate
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            # Compute bias-corrected second raw moment estimate
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)
            # Update parameter with weight decay incorporated
            parameters[key] -= self.learning_rate * (m_hat + self.weight_decay * parameters[key]) / (np.sqrt(v_hat) + self.epsilon)

class Nadam:
    def __init__(self, parameters, learning_rate=1e-3, beta1=0.9, beta2=0.99, epsilon=1e-6, weight_decay=0.0):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.t = 0
        # Initialize first and second moment estimates for each parameter
        self.m = {key: np.zeros_like(val) for key, val in parameters.items()}
        self.v = {key: np.zeros_like(val) for key, val in parameters.items()}

    def update(self, parameters, grads):
        self.t += 1
        for key in parameters.keys():
            grad = grads.get("d" + key, 0)
            # Update biased first moment estimate
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grad
            # Update biased second raw moment estimate
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grad ** 2)
            # Bias correction for first moment
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            # Bias correction for second moment
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)
            # Nadam update: combines Nesterov momentum with Adam
            parameters[key] -= self.learning_rate * (
                (self.beta1 * m_hat + (1 - self.beta1) * grad / (1 - self.beta1 ** self.t)) 
                + self.weight_decay * parameters[key]
            ) / (np.sqrt(v_hat) + self.epsilon)


optimizer_dict = {
    "sgd": SGD,
    "momentum": Momentum,
    "nag": NAG,
    "rmsprop": RMSprop,
    "adam": Adam,
    "nadam": Nadam
}

def get_optimizer(opt_name, parameters, **kwargs):
    """
    Factory function to create an optimizer.
    Arguments:
      opt_name    : Name of the optimizer (e.g., "adam", "nag").
      parameters  : Dictionary of parameters (weights and biases) of the model.
      kwargs      : Additional keyword arguments (learning_rate, momentum, etc.).
    Returns:
      An instance of the selected optimizer.
    """
    if opt_name not in optimizer_dict:
        raise ValueError(f"Unsupported optimizer '{opt_name}'")
    
    optimizer_class = optimizer_dict[opt_name]
    
    # Get the signature of the __init__ method, ignoring 'self' and 'parameters'
    sig = inspect.signature(optimizer_class.__init__)
    accepted_params = set(sig.parameters.keys()) - {'self', 'parameters'}
    
    # Filter kwargs to include only those accepted by the optimizer
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in accepted_params}
    
    return optimizer_class(parameters, **filtered_kwargs)
    