import re
import argparse
import numpy as np
import wandb
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist, mnist
from sklearn.metrics import confusion_matrix

from src.model import NeuralNetwork
from src.utils import one_hot_encode, plot_confusion_matrix

def extract_config_from_filename(filename):
    """
    Extract hyperparameters from the weights filename.
    Expected filename format:
    fhl_{num_layers}_hs_{hidden_size}_bs_{batch_size}_ep_{epochs}_ac_{activation}_o_{optimizer}_lr_{learning_rate}_wd_{weight_decay}_wi_{weight_init}_dataset_{dataset}_loss_{loss}.npy
    """
    pattern = (
        r"fhl_(?P<num_layers>\d+)_hs_(?P<hidden_size>\d+)_bs_(?P<batch_size>\d+)_ep_(?P<epochs>\d+)_"
        r"ac_(?P<activation>[^_]+)_o_(?P<optimizer>[^_]+)_lr_(?P<learning_rate>[^_]+)_"
        r"wd_(?P<weight_decay>[^_]+)_wi_(?P<weight_init>[^_]+)_dataset_(?P<dataset>[^_]+)_loss_(?P<loss>[^\.]+)"
    )
    m = re.search(pattern, filename)
    if m:
        config = m.groupdict()
        # Convert numeric values to appropriate types
        config['num_layers'] = int(config['num_layers'])
        config['hidden_size'] = int(config['hidden_size'])
        config['batch_size'] = int(config['batch_size'])
        config['epochs'] = int(config['epochs'])
        config['learning_rate'] = float(config['learning_rate'])
        config['weight_decay'] = float(config['weight_decay'])
        return config
    else:
        raise ValueError("Could not parse configuration from filename.")

def evaluate(weights_file):
    """
    Evaluates the trained model given the model filename.
    
    Args:
        weights_file (str): Path to the saved model weights (.npy file)
    
    Returns:
        test_loss, test_accuracy, confusion_matrix
    """
    # Extract configuration from the filename
    config_from_file = extract_config_from_filename(weights_file)
    
    # Initialize wandb with the extracted configuration
    wandb.init(project="DA6401_Assignment1",
               entity="ns25z040-indian-institute-of-technology-madras",
               config=config_from_file)
    config = wandb.config
    run_name = f"eval_fhl_{config.num_layers}_hs_{config.hidden_size}_ac_{config.activation}_loss_{config.loss}_dataset_{config.dataset}"
    wandb.run.name = run_name
    wandb.run.save()

    # Load the test dataset based on the extracted dataset name
    if config.dataset == "fashion_mnist":
        (_, _), (X_test, y_test) = fashion_mnist.load_data()
    else:
        (_, _), (X_test, y_test) = mnist.load_data()

    # Preprocess test data: flatten and normalize images
    X_test = X_test.reshape(X_test.shape[0], -1) / 255.0
    num_classes = 10
    y_test_encoded = one_hot_encode(y_test, num_classes)

    # Initialize the Neural Network with the extracted configuration
    input_size = X_test.shape[1]
    output_size = num_classes
    nn = NeuralNetwork(input_size=input_size,
                       hidden_size=config.hidden_size,
                       num_layers=config.num_layers,
                       output_size=output_size,
                       activation=config.activation,
                       weight_init=config.weight_init,
                       loss_type=config.loss)

    # Load the saved model weights (using .item() to extract the dictionary)
    best_weights = np.load(weights_file, allow_pickle=True).item()
    nn.parameters = best_weights

    # Evaluate the model on the test set
    test_outputs = nn.forward(X_test)
    test_loss = nn.compute_loss(test_outputs, y_test_encoded)
    test_predictions = np.argmax(test_outputs, axis=1)
    test_accuracy = np.mean(test_predictions == y_test)

    # Log test metrics to wandb
    wandb.log({
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
    })

    # Print evaluation results
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Compute and plot confusion matrix
    cm = confusion_matrix(y_test, test_predictions)
    if config.dataset == "fashion_mnist":
        labels = [
            "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
        ]
    else:
        labels = [str(i) for i in range(num_classes)]
    plot_confusion_matrix(cm, labels, run_name="Test Confusion Matrix", project="DA6401_Assignment1")
    plt.show()

    return test_loss, test_accuracy, cm

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Feedforward Neural Network on Fashion-MNIST or MNIST"
    )
    parser.add_argument('-f', '--weights_file', type=str, required=True,
                        help="Path to the saved model weights (.npy file)")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_arguments()
    evaluate(args.weights_file)
