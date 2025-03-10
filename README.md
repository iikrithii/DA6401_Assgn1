# DA6401 - Introduction to Deep Learning
## Assignment 1: Feedforward Neural Network for Fashion-MNIST

This repository contains the complete source code, notebooks, and reports for an assignment where a custom feedforward neural network is implemented from scratch. The project demonstrates how to build a neural network without relying on automatic differentiation libraries and emphasizes modularity, hyperparameter tuning, and experiment tracking via Weights & Biases (Wandb). The detailed report on the implementation is recorded on Wandb and can be accessed here ([Guide to Report](https://wandb.ai/ns25z040-indian-institute-of-technology-madras/DA6401_Assignment1/reports/DA6401-Assignment-1--VmlldzoxMTY5NzMxNQ?accessToken=6mfx6g8axqfkjbacx2umk9ln0a0cw43hwbwwf998bseq81wtk0dns8pxt0bj66sy)).

---

## Table of Contents

- [Overview](#overview)
    - [Problem Statement](#problem-statement)
    - [Deliverables](#deliverables)
- [File Structure](#file-structure)
- [Implementation](#implementation)
  - [Question 1: Data Visualization](#question-1-data-visualization)
  - [Question 2: Feedforward Neural Network](#question-2-feedforward-neural-network)
  - [Question 3: Backpropagation and Optimization](#question-3-backpropagation-and-optimization)
  - [Question 4: Implementing Neural Network](#question-4-implementing-neural-network)
  - [Question 5-6: Conducting Sweeps](#question-5-6-conducting-sweeps)
  - [Question 7: Evaluation](#question-7-evaluation)

- [Command-Line Arguments](#command-line-arguments)


- [Scalability and Reproducability](#scalability-and-reproducability)
    - [Training your own model](#training-your-own-model)
  - [Adding New Loss Functions](#adding-new-loss-functions)
  - [Adding New Activation Functions](#adding-new-activation-functions)
  - [Adding New Optimizers](#adding-new-optimizers)
- [Results and Final Hyperparameters](#results-and-final-hyperparameters)
- [Conclusion](#conclusion)

---
## Overview

This repository is part of an assignment in which a fully-functional feedforward neural network is implemented and trained on the Fashion-MNIST dataset.

### Problem Statement

The primary goal of this assignment is to implement a classification model for the Fashion-MNIST dataset using a feedforward neural network built from scratch and extending it to the following:

- **Implementing Backpropagation:** Develop a custom backpropagation algorithm to update network weights based on gradients computed manually.

- **Supporting Multiple Optimizers:** Implement several optimization techniques (SGD, Momentum, Nesterov, RMSprop, Adam, Nadam) to compare their effects on convergence and accuracy.

- **Hyperparameter Tuning:** Use systematic hyperparameter sweeps (tracked via Wandb) to find optimal model configurations, including the number of layers, learning rate, batch size, activation function, weight initialization, and more.

- **Evaluation:** Evaluate the trained model on unseen test data, with detailed performance analysis through confusion matrices and other visualizations.

The approach and results of this implementation are further detailed in this [report](https://wandb.ai/ns25z040-indian-institute-of-technology-madras/DA6401_Assignment1/reports/DA6401-Assignment-1--VmlldzoxMTY5NzMxNQ/edit?draftId=VmlldzoxMTcxMjAzNw==). This repository serves a code source that implements the following:

### Deliverables 
- **Modular Code Design:** All components (data preprocessing, model, activations, losses, optimizers, and utilities) are separated into individual modules, making it easy to extend the project.

- **Custom Implementation:** The network is implemented without relying on automatic differentiation libraries, giving full control over forward and backward computations.

- **Experiment Tracking:** All experiments, hyperparameter sweeps, and model evaluations are logged using Wandb to ensure reproducibility and transparency.

- **Hyperparameter Optimization:** Extensive hyperparameter sweeps (via Wandb) are used to identify the best model configurations.

- **Evaluation and Visualization:** The repository includes scripts for both training and evaluation, with detailed visualizations (such as sample images and confusion matrices) to analyze model performance.

## File Structure

Below is an overview of the repository structure, which is organized to enhance modularity and ease of navigation:


---
```bash
├── Q1_Sample.ipynb             
# Notebook for visualizing sample images from the Fashion-MNIST dataset.
├── Q2_Feedforward.ipynb        
# Notebook for running a forward pass on the network and visualizing prediction probabilities.
├── train.py                    
# Main training script that orchestrates data loading, model training, logging, and model saving.
├── evaluate.py                 
# Script to evaluate a saved model on the test set and generate performance metrics.
├── models/                     
# Directory for storing saved model files corresponding to the best performing configurations.
├── sweep/                      
# Folder containing sample configuration files for Wandb hyperparameter sweeps.
├── requirements.txt
# Contains all the required packages for running this project in an environment.
└── src/                        
    ├── activation.py           
    # Contains all activation functions (sigmoid, tanh, ReLU, identity) and their derivatives.
    ├── loss.py                 
    # Contains loss functions (cross-entropy and mean squared error) with their gradient implementations.
    ├── model.py                
    # Implements the NeuralNetwork class, which includes methods for the forward and backward passes.
    ├── optimizers.py           
    # Contains implementations of different optimization algorithms (SGD, momentum, NAG, RMSprop, Adam, Nadam).
    └── utils.py                
    # Utility functions for one-hot encoding, image plotting, and visualization of confusion matrices.
```

## Implementation
### Question 1: Data Visualization

[`Q1_Sample.ipynb` ](Q1_Sample.ipynb) visualises samples from the Fashion-MNIST dataset using the following approach. 
  - **Data Loading:** The dataset is loaded using Keras’ `fashion_mnist.load_data()` function.
  - **Image Display:** A grid of 25 sample images (one per class) is plotted using Matplotlib, where each image is labeled with its corresponding class name.
  - **Wandb Logging:** The plotted grid is logged to Wandb for real-time visualization and tracking. 

  
### Question 2: Feedforward Neural Network

[`Q2_Feedforward.ipynb`](Q2_Feedforward.ipynb) serves as a testing ground for the network:
  - **Model Initialization:** The neural network is instantiated from the `NeuralNetwork` class defined in [`src/model.py`](src/model.py)
  - **Forward Pass:** A forward pass is performed on a subset of test images to generate probability distributions over classes.
  - **Visualization:** Each sample’s prediction probabilities are visualized as bar plots, annotated with their numerical values.
  - **Wandb Logging:** All visual outputs are logged to Wandb.

### Question 3: Backpropagation and Optimization

- **Model Implementation [`src/model.py`](src/model.py)**  
  The `NeuralNetwork` class is the core of the model configuration:
  - **Forward Pass:** The `forward_pass()` method computes intermediate activations and the final output. It supports both softmax (for cross-entropy loss) and linear outputs (for mean squared error).
  - **Backward Pass:** The `backward_pass()` method calculates the gradients for all layers by applying the chain rule using the derivatives of the activation and loss functions.
  - **Forward:** The `forward()` method simply returns only the probabilities for the function to evaluate a simple forward network to output the final values/probabilities. 

These functions have been modularized to work with varying input arguments for batch sizes, hidden layers, neuron sizes, optimzers, activations and so on. This ensures the scalability and generalisability of these functions during implementation. 

- **Optimizers [`src/optimizers.py`](src/optimizers.py):**  
  Several optimizers have been implemented:
  - **SGD:** Basic stochastic gradient descent.
  - **Momentum & Nesterov:** Incorporates momentum to speed up convergence and overcome local minima.
  - **RMSprop:** Adapts the learning rate per parameter.
  - **Adam & Nadam:** Uses adaptive learning rates and, in the case of Nadam, incorporates Nesterov momentum.

- **Activation Functions [`src/activation.py`](src/activation.py) & Loss Functions [`src/loss.py`](src/loss.py):**  
  The includes implementations for commonly used activation functions (sigmoid, tanh, ReLU, identity) and their derivatives. Additionally, it provides two loss functions (cross-entropy and mean squared error) with their gradient calculations.

### Question 4: Implementing Neural Network

1. **Data Loading and Preprocessing:**  
   - **Dataset:** The Fashion-MNIST dataset is loaded using Keras.
   - **Preprocessing:** Images are flattened into vectors and normalized. Labels are one-hot encoded.
   - **Data Splitting:** The training data is split into a training set (90%) and a validation set (10%).

2. **Model Initialization:**  
   - The `NeuralNetwork` class is instantiated with user-defined hyperparameters (number of layers, hidden size, activation function, etc.).
   - Model parameters (weights and biases) are initialized using either Xavier or a random method.

3. **Training Loop:**  
   - **Shuffling and Batching:** Each epoch begins with shuffling the training data and dividing it into mini-batches.
   - **Forward Pass:** The network’s `forward_pass()` computes the output probabilities and intermediate activations.
   - **Loss Computation:** The loss is calculated using the selected loss function.
   - **Backward Pass:** The `backward_pass()` computes gradients using the chain rule.
   - **Parameter Updates:** The optimizer’s `update()` function adjusts the model parameters based on the computed gradients.
   - **Metrics Logging:** Training and validation losses, accuracies, and other metrics are logged to Wandb for real-time tracking.

4. **Saving the Best Model:**  
   - If the current epoch’s validation accuracy exceeds the best recorded value, the model’s parameters are saved to the `models/` directory. The saved filename encodes the hyperparameter configuration for easy reference during evaluation.

### Question 5-6: Conducting Sweeps

Several sweeps were conducted with different configurations ranging from a broader search to a more narrowed analysis using the best hyperparameters from previous run. The hyperparameter sweep configurations are stored in [`sweeps/`](sweeps/) folder. 

### Question 7: Evaluation 

The main evaluation is loaded onto the training loop that uses the `plot_confusion_matrix` function from [`utils.py`](src/utils.py) to test the accuracy on the trained model and plot an interactive confusion matrix. 

In addition `evaluate.py` script is designed to assess the performance of a saved model, that helps test a model's configuration with just the best model saved using the train function. 

- **Extract Configuration:** The script extracts hyperparameter settings from the saved model filename.
- **Data Preparation:** The test dataset is loaded, preprocessed, and one-hot encoded.
- **Model Reconstruction:** The network is reconstructed using the extracted hyperparameters.
- **Loading Weights:** The saved model parameters are loaded into the network.
- **Performance Metrics:** A forward pass is executed on the test data to compute test loss, accuracy, and generate a confusion matrix.
- **Wandb Logging:** All evaluation metrics and visualizations are logged to Wandb.

---

## Command-Line Arguments

The `train.py` script uses Python’s `argparse` to define a set of command-line arguments that control various aspects of training. Each parameter has a clear prefix and default value. 

| Prefix(es)                                             | Description                                                                                       | Default Value           |
|--------------------------------------------------------|---------------------------------------------------------------------------------------------------|-------------------------|
| `-wp`, `--wandb_project`                                | Name of the Wandb project.                                                                        | *myprojectname*        |
| `-we`, `--wandb_entity`                                 | Wandb entity (username or team).                                                                  | *myname*        |
| `-d`, `--dataset`                                      | Choice between `"mnist"` and `"fashion_mnist"`.                                                 | `"fashion_mnist"`       |
| `-e`, `--epochs`                                       | Number of training epochs.                                                                        | `1`                    |
| `-b`, `--batch_size`                                   | Batch size for training.                                                                          | `4`                    |
| `-l`, `--loss`                                         | Loss function to use; options are `"mean_squared_error"` or `"cross_entropy"`.                    | `"cross_entropy"`       |
| `-o`, `--optimizer`                                    | Optimizer choice; options include `"sgd"`, `"momentum"`, `"nag"`, `"rmsprop"`, `"adam"`, `"nadam"`. | `"sgd"`                |
| `-lr`, `--learning_rate`                               | Learning rate for the optimizer.                                                                  | `0.1`                  |
| `-m`, `--momentum`                                     | Momentum factor (for momentum and NAG optimizers).                                                | `0.5`                   |
| `--beta`                                             | Beta parameter for RMSprop.                                                                       | `0.5`      |
| `--beta1`                                             | Beta1 for Adam and Nadam optimizers.                                                              | `0.5`                   |
| `--beta2`                                             | Beta2 for Adam and Nadam optimizers.                                                              | `0.5`                   |
| `-eps`, `--epsilon`                                    | Epsilon for numerical stability in optimizers.                                                  | `1e-6`                  |
| `-w_d`, `--weight_decay`                               | L2 regularization factor.                                                                         | `0.0`                   |
| `-w_i`, `--weight_init`                                | Weight initialization method; options are `"random"` or `"Xavier"`.                               | `"random"`              |
| `-nhl`, `--num_layers`                                 | Number of hidden layers in the network.                                                           | `1`                     |
| `-sz`, `--hidden_size`                                 | Number of neurons per hidden layer.                                                               | `4`                    |
| `-a`, `--activation`                                   | Activation function; options are `"identity"`, `"sigmoid"`, `"tanh"`, `"ReLU"`.                   | `"sigmoid"`                |
| `-ev`, `--evaluate`                                    | Boolean flag to run test evaluation after training.                                             | `True`                  |


This detailed argument structure allows users to flexibly experiment with different model configurations and training settings.

---

## Scalability and Reproducability

This project is built with modularity in mind, making it straightforward to extend the functionality. New components can be added without disrupting the existing code structure. Below are details on how to reproduce the code, train your own network, and the system in terms of loss functions, activation functions, and optimizers.

### Training your own Model

Before training the model, install the required Python packages by running:

```bash
pip install -r requirements.txt
```

You can then define your own model parameter configurations based on the arguments as defined in the [command line arguments](#command-line-arguments). Here is a template for the same:

```bash
python train.py -wp "project-name" -d "dataset" -e epochs -b <batch_size> -l "loss" -o "optimizer" -lr <learning_rate> -nhl <num_layers> -sz <hidden_size> -a "activation" -ev <evaluation>
```

### Adding New Loss Functions


Add the following code to [`src/loss.py`](src/loss.py):

```python
def new_loss(y_pred, y_true, <other_hyperparameters>)
    """
    Add the loss function equation and return the value of error
    """
def new_loss_deriv(y_pred, y_true, <other_hyperparameters>)
    """
    Add the loss function derivative to compute during the backpropagation
    """
```

Update the loss_functions dictionary in `src/loss.py` by adding an entry:

```python
loss_functions = {
    "cross_entropy": (cross_entropy, cross_entropy_deriv),
    "mean_squared_error": (mean_squared_error, mean_squared_error_deriv),
    "new_loss": (new_loss, new_loss_deriv)
}
```

### Adding New Activation Functions

In [`src/activation.py`](src/activation.py), add your custom activation function and its derivative. For example:

```python
def new_activation(z):
    """
    Add the custom activation equation of input z.
    
    Return the post activation value.
    """

def new_activation_deriv(z):
    """
    Add the derivative of the custom activation function.
    
    Return the derivative of the activation for backpropagation.
    """
```
Update the activations dictionary

```python
activations = {
    "sigmoid": (sigmoid, sigmoid_derivative),
    "tanh": (tanh, tanh_derivative),
    "ReLU": (relu, relu_derivative),
    "identity": (identity, identity_derivative),
    "new_activation": (new_activation, new_activation_deriv)
}
```

### Adding New Optimizers

To extend the repository with a new optimizer, follow these generic steps:

1. **Define a New Optimizer Class**

   In [`src/optimizers.py`](src/optimizers.py), create a new class for your optimizer. The class should include:
   
   - An initializer (`__init__`) that sets up the required hyperparameters and initializes any state variables (e.g., momentum terms, moving averages).
   - An `update()` method that implements the parameter update rule based on the computed gradients.

   ```python
   class NewOptimizer:
       """
       A generic template for a new optimizer.
       Replace the update rule with your custom logic.
       """
       def __init__(self, parameters, learning_rate, **kwargs):
           self.learning_rate = learning_rate
           # Initialize any necessary state variables here.
           self.state = {key: np.zeros_like(val) for key, val in parameters.items()}

       def update(self, parameters, grads):
           # Implement your custom update logic for each parameter.
           # Update self.state and adjust the parameters accordingly.
           pass
    ```

Update the helper function (e.g., `get_optimizer`) in `src/optimizers.py` to include your new optimizer. Add a condition that checks for your optimizer’s identifier and returns an instance of your new optimizer.

```python
def get_optimizer(optimizer_name, parameters, learning_rate, momentum, beta, beta1, beta2, epsilon, weight_decay):
    if optimizer_name == "sgd":
        # Existing optimizer...
        pass
    elif optimizer_name == "new_optimizer":
        return NewOptimizer(parameters, learning_rate)
    else:
        raise ValueError("Unsupported optimizer")
```

After integration, you can select your new optimizers, activations or losses via the command-line argument (e.g., -o new_optimizer) when running the training script.

---
## Results and Final Hyperparameters

After extensive hyperparameter tuning using systematic Wandb sweeps, the optimal configuration for the Fashion-MNIST classification task was determined to be:

- **Epochs:** 30  
- **Batch Size:** 256  
- **Number of Hidden Layers:** 3  
- **Hidden Layer Size:** 256  
- **Activation Function:** tanh  
- **Optimizer:** nadam  
- **Learning Rate:** 1e-3  
- **Weight Decay:** 0  
- **Weight Initialization:** Xavier  
- **Loss Function:** cross_entropy  
- **Additional Optimizer Parameters:**  
  - Beta1 = 0.9  
  - Beta2 = 0.999  

This configuration achieved a **validation accuracy of approximately 90.2%** and a **test accuracy of approximately 89.48%**.

---

## Conclusion

The experimental results validate the effectiveness of the implemented feedforward neural network architecture and the systematic approach to hyperparameter tuning to experiment with a well established dataset and familiarise with different types of hyperparameter tuning.




