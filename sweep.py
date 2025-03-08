import pprint
import wandb
from train import main 

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
    # 'early_terminate': {
    #     'type': 'hyperband',
    #     'min_iter': 3,  
    #     'max_iter': 10,  
    # },
    'epochs': {
        'values': [5, 10]
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
        'values': ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"]
    },
    'batch_size': {
        'values': [16, 32, 64, 128]
    },
    'weight_init': {
        'values': ["random", "Xavier"]
    },
    'activation': {
        'values': ["sigmoid", "tanh", "ReLU"]
    },
}
sweep_config['parameters'] = parameters_dict
pprint.pprint(sweep_config)
sweep_id= wandb.sweep(sweep_config, project="DA6401_Assignment1")
print(sweep_id)
wandb.agent(sweep_id, function=main)