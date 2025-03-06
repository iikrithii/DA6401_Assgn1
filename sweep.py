# Define the sweep configuration 
sweep_config = {
    'method': 'bayes',
}

# Define the metric configuration
metric = {
    'name': 'val_accuracy',
    'goal': 'maximize'
}
sweep_config['metric'] = metric
    
# Define the hyperparameters (parameters) dictionary
parameters_dict = {
    'epochs': {
        'values': [15]
    },
    'num_layers': {
        'values': [3, 4, 5]
    },
    'hidden_size': {
        'values': [32, 64, 128]
    },
    'weight_decay': {
        'values': [0, 0.0005, 0.5]
    },
    'learning_rate': {
        'values': [1e-3, 1e-4]
    },
    'optimizer': {
        'values': ["sgd", "momentum", "nag", "rmsprop"]
    },
    'batch_size': {
        'values': [16, 32, 64]
    },
    'weight_init': {
        'values': ["random", "Xavier"]
    },
    'activation': {
        'values': ["sigmoid", "tanh", "ReLU"]
    },
}
sweep_config['parameters'] = parameters_dict



import pprint
import wandb
from train import main 


pprint.pprint(sweep_config)
sweep_id= wandb.sweep(sweep_config, project="Assgn_1")
wandb.agent(sweep_id, function=main)