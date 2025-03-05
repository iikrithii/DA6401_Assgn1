# train.py
import argparse
import wandb
import numpy as np
from keras.datasets import fashion_mnist, mnist
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import os

from src.model import NeuralNetwork
from src.utils import one_hot_encode
from src.optimizers import get_optimizer

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train a Feedforward Neural Network on Fashion-MNIST or MNIST")
    parser.add_argument('-wp', '--wandb_project', type=str, default="myprojectname", help="Project name used to track experiments in Weights & Biases dashboard")
    parser.add_argument('-we', '--wandb_entity', type=str, default="myname", help="Wandb Entity used to track experiments in Weights & Biases dashboard")
    parser.add_argument('-d', '--dataset', type=str, choices=["mnist", "fashion_mnist"], default="fashion_mnist", help="Dataset to use")
    parser.add_argument('-e', '--epochs', type=int, default=10, help="Number of epochs to train neural network")
    parser.add_argument('-b', '--batch_size', type=int, default=32, help="Batch size used to train neural network")
    parser.add_argument('-l', '--loss', type=str, choices=["mean_squared_error", "cross_entropy"], default="cross_entropy", help="Loss function to use")
    parser.add_argument('-o', '--optimizer', type=str, choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"], default="adam", help="Optimizer to use")
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3, help="Learning rate used to optimize model parameters")
    parser.add_argument('-m', '--momentum', type=float, default=0.5, help="Momentum used by momentum and nag optimizers")
    parser.add_argument('--beta', type=float, default=0.5, help="Beta used by rmsprop optimizer")
    parser.add_argument('--beta1', type=float, default=0.5, help="Beta1 used by adam and nadam optimizers")
    parser.add_argument('--beta2', type=float, default=0.5, help="Beta2 used by adam and nadam optimizers")
    parser.add_argument('-eps', '--epsilon', type=float, default=1e-6, help="Epsilon used by optimizers")
    parser.add_argument('-w_d', '--weight_decay', type=float, default=0.0, help="Weight decay used by optimizers")
    parser.add_argument('-w_i', '--weight_init', type=str, choices=["random", "Xavier"], default="Xavier", help="Weight initialization method")
    parser.add_argument('-nhl', '--num_layers', type=int, default=3, help="Number of hidden layers used in feedforward neural network")
    parser.add_argument('-sz', '--hidden_size', type=int, default=64, help="Number of neurons in each hidden layer")
    parser.add_argument('-a', '--activation', type=str, choices=["identity", "sigmoid", "tanh", "ReLU"], default="ReLU", help="Activation function to use")
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()

    # Initialize wandb
    run_name = f"hl_{config.num_layers}_bs_{config.batch_size}_ac_{config.activation}"
    wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args), name=run_name)
    config = wandb.config

    # Load dataset
    if config.dataset == "fashion_mnist":
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    else:
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Preprocess: flatten images and normalize pixel values
    X_train = X_train.reshape(X_train.shape[0], -1) / 255.0
    X_test = X_test.reshape(X_test.shape[0], -1) / 255.0

    # Split training data: 90% train, 10% validation
    split_index = int(0.9 * X_train.shape[0])
    X_val = X_train[split_index:]
    y_val = y_train[split_index:]
    X_train = X_train[:split_index]
    y_train = y_train[:split_index]

    # One-hot encode labels
    num_classes = 10
    y_train_encoded = one_hot_encode(y_train, num_classes)
    y_val_encoded = one_hot_encode(y_val, num_classes)
    y_test_encoded = one_hot_encode(y_test, num_classes)

    # Question 2: Initialize Neural Network
    input_size = X_train.shape[1]
    output_size = num_classes
    nn = NeuralNetwork(input_size=input_size,
                       hidden_size=config.hidden_size,
                       num_layers=config.num_layers,
                       output_size=output_size,
                       activation=config.activation,
                       weight_init=config.weight_init,
                       loss_type=config.loss)
    
    # Initialize optimizer
    optimizer = get_optimizer(config.optimizer,
                              nn.parameters,
                              learning_rate=config.learning_rate,
                              momentum=config.momentum,
                              beta=config.beta,
                              beta1=config.beta1,
                              beta2=config.beta2,
                              epsilon=config.epsilon,
                              weight_decay=config.weight_decay)

    num_batches = int(np.ceil(X_train.shape[0] / config.batch_size))
   