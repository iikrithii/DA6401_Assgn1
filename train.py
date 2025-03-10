# train.py
import argparse
import wandb
import numpy as np
from keras.datasets import fashion_mnist, mnist
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import os

from src.model import NeuralNetwork
from src.utils import one_hot_encode, plot_confusion_matrix
from src.optimizers import get_optimizer

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train a Feedforward Neural Network on Fashion-MNIST or MNIST",
                                     allow_abbrev=False)
    parser.add_argument('-wp', '--wandb_project', type=str, default="myprojectname", help="Project name used to track experiments in Weights & Biases dashboard")
    parser.add_argument('-we', '--wandb_entity', type=str, default="myname", help="Wandb Entity used to track experiments in Weights & Biases dashboard")
    parser.add_argument('-d', '--dataset', type=str, choices=["mnist", "fashion_mnist"], default="fashion_mnist", help="Dataset to use")
    parser.add_argument('-e', '--epochs', type=int, default=1, help="Number of epochs to train neural network")
    parser.add_argument('-b', '--batch_size', type=int, default=4, help="Batch size used to train neural network")
    parser.add_argument('-l', '--loss', type=str, choices=["mean_squared_error", "cross_entropy"], default="cross_entropy", help="Loss function to use")
    parser.add_argument('-o', '--optimizer', type=str, choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"], default="sgd", help="Optimizer to use")
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.1, help="Learning rate used to optimize model parameters")
    parser.add_argument('-m', '--momentum', type=float, default=0.5, help="Momentum used by momentum and nag optimizers")
    parser.add_argument('--beta', type=float, default=0.5, help="Beta used by rmsprop optimizer")
    parser.add_argument('--beta1', type=float, default=0.5, help="Beta1 used by adam and nadam optimizers")
    parser.add_argument('--beta2', type=float, default=0.5, help="Beta2 used by adam and nadam optimizers")
    parser.add_argument('-eps', '--epsilon', type=float, default=1e-6, help="Epsilon used by optimizers")
    parser.add_argument('-w_d', '--weight_decay', type=float, default=0.0, help="Weight decay used by optimizers")
    parser.add_argument('-w_i', '--weight_init', type=str, choices=["random", "Xavier"], default="random", help="Weight initialization method")
    parser.add_argument('-nhl', '--num_layers', type=int, default=1, help="Number of hidden layers used in feedforward neural network")
    parser.add_argument('-sz', '--hidden_size', type=int, default=4, help="Number of neurons in each hidden layer")
    parser.add_argument('-a', '--activation', type=str, choices=["identity", "sigmoid", "tanh", "ReLU"], default="sigmoid", help="Activation function to use")
    parser.add_argument('-ev', '--evaluate', type=bool, default=1, help="Test on data and report test accuracy")
    args, unknown = parser.parse_known_args()
    return args

def main():
    args = parse_arguments()
    # Initialize wandb
    wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args))
    config = wandb.config

    run_name = f"hl_{config.num_layers}_hs_{config.hidden_size}_bs_{config.batch_size}_ep_{config.epochs}_ac_{config.activation}_o_{config.optimizer}_lr_{config.learning_rate}_wd_{config.weight_decay}_wi_{config.weight_init}_dataset_{config.dataset}"
    wandb.run.name = run_name
    wandb.run.save()

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
   
    best_val_accuracy = 0.0
    best_epoch= 0
    best_train_accuracy=0.0
    best_train_loss= np.inf
    best_val_loss= np.inf

    # Training loop
    for epoch in range(config.epochs):
        # Shuffle training data
        permutation = np.random.permutation(X_train.shape[0])
        X_train = X_train[permutation]
        y_train_encoded = y_train_encoded[permutation]

        epoch_loss = 0.0
        correct = 0

        for i in range(num_batches):
            start = i * config.batch_size
            end = start + config.batch_size
            X_batch = X_train[start:end]
            y_batch = y_train_encoded[start:end]

            # Forward pass
            a_list, h_list, outputs = nn.forward_pass(X_batch)

            # Compute loss
            loss = nn.compute_loss(outputs, y_batch)
            epoch_loss+=loss

            # Backward pass
            grads = nn.backward_pass(a_list, h_list, y_batch, outputs)

            # Update parameters
            optimizer.update(nn.parameters, grads)

            # Compute training accuracy for this batch
            predictions = np.argmax(outputs, axis=1)
            labels = np.argmax(y_batch, axis=1)
            correct += np.sum(predictions == labels)

        train_accuracy = correct / X_train.shape[0]
        val_outputs= nn.forward(X_val)
        val_loss = nn.compute_loss(val_outputs, y_val_encoded)
        val_predictions = np.argmax(val_outputs, axis=1)
        val_accuracy = np.mean(val_predictions == y_val)    

        # Log metrics to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": epoch_loss / num_batches,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        })

        print(f"Epoch {epoch + 1}/{config.epochs} - Train Loss: {epoch_loss / num_batches:.4f}, Train Acc: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
    
        # Save best model 
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_train_accuracy= train_accuracy
            best_epoch= epoch
            best_train_loss= epoch_loss / num_batches
            best_val_loss= val_loss
            filename = f"models/hl_{config.num_layers}_hs_{config.hidden_size}_bs_{config.batch_size}_ep_{config.epochs}_ac_{config.activation}_o_{config.optimizer}_lr_{config.learning_rate}_wd_{config.weight_decay}_wi_{config.weight_init}_dataset_{config.dataset}_loss_{config.loss}.npy"
            np.save(filename, nn.parameters)

    # Log metrics to wandb
    wandb.log({
        "best_val_accuracy": best_val_accuracy,
        "best_epoch": best_epoch,
        "best_train_loss": best_train_loss,
        "best_train_accuracy": best_train_accuracy,
        "best_val_loss": best_val_loss,
    })
    
    if config.evaluate:
        best_weights = np.load(filename, allow_pickle=True).item()
        nn.parameters = best_weights

        # Compute test accuracy with the best model
        test_outputs = nn.forward(X_test)
        test_loss = nn.compute_loss(test_outputs, y_test_encoded)
        test_predictions = np.argmax(test_outputs, axis=1)
        test_accuracy = np.mean(test_predictions == y_test)

        wandb.log({
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        })

        cm = confusion_matrix(y_test, test_predictions)

        if config.dataset == "fashion_mnist":
            labels = [
            "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
            ]
        else:
            num_classes = cm.shape[0]
            labels = [str(i) for i in range(num_classes)]

        plot_confusion_matrix(cm, labels, run_name="Test Confusion Matrix", project= args.wandb_project)

if __name__ == "__main__":
    main()

# 
