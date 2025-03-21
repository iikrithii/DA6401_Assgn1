import pprint
import wandb
from train import main 

# Define the sweep configuration 
sweep_config = {
    'method': 'bayes',
    'early_terminate': {
        'type': 'hyperband',
        'min_iter': 7,  
        'max_iter': 20,  
        'eta': 2,
    },
}

# Define the metric configuration
metric = {
    'name': 'best_val_accuracy',
    'goal': 'maximize'
}
sweep_config['metric'] = metric
    
# Define the hyperparameters (parameters) dictionary
parameters_dict = {
    'epochs': {
        'values': [20]
    },
    'num_layers': {
        'values': [3, 4, 5]
    },
    'hidden_size': {
        'values': [32, 64, 128]
    },
    'weight_decay': {
        'values': [0]
    },
    'learning_rate': {
        'values': [1e-3, 1e-4]
    },
    'optimizer': {
        'values': ["rmsprop", "adam", "nadam"]
    },
    'batch_size': {
        'values': [64, 128, 256]
    },
    'weight_init': {
        'values': ["Xavier"]
    },
    'activation': {
        'values': ["tanh", "ReLU"]
    },
    'beta1': {
        'values': [0.5, 0.9]
    },
    'beta2': {
        'values': [0.5, 0.999]
    },

    
}
sweep_config['parameters'] = parameters_dict
pprint.pprint(sweep_config)
sweep_id= wandb.sweep(sweep_config, project="DA6401_Assignment1")
print(sweep_id)
wandb.agent(sweep_id, function=main)