batch_size = 64

config = {}

# set the parameters related to the training and testing set
data_train_opt = {} 
data_train_opt['batch_size'] = batch_size
data_train_opt['dataset_name'] = 'action_box_stacking_hard_task_MM'
data_train_opt['split'] = 'train'
data_train_opt['dtype'] = 'latent'
data_train_opt['img_size'] = 256
data_train_opt['task_name'] = 'unity_stacking'

data_test_opt = {}
data_test_opt['batch_size'] = batch_size
data_test_opt['epoch_size'] = None
data_test_opt['dataset_name'] = 'action_box_stacking_hard_task_MM'
data_test_opt['split'] = 'test'
data_test_opt['dtype'] = 'latent'
data_test_opt['img_size'] = 256
data_test_opt['task_name'] = 'unity_stacking'

data_original_opt = {}
data_original_opt['path_to_original_dataset'] = './datasets/'
data_original_opt['path_to_original_train_data'] = './datasets/action_data/train_action_box_stacking_hard_task_2500'
data_original_opt['path_to_original_test_data'] = './datasets/action_data/test_action_box_stacking_hard_task_2500'

config['data_train_opt'] = data_train_opt
config['data_test_opt'] = data_test_opt
config['data_original_opt'] = data_original_opt

model_opt = {
    'filename': 'apnet', 
    'vae_name': 'box_stacking_hard_task_MM',
    'model_module': 'APM',
    'model_class': 'APNet',

    'device': 'cuda',
    'dims': [12*2, 100, 100, 100, 5],
    'dropout': 0,

    'epochs': 500,
    'batch_size': batch_size,
    'lr_schedule': [(0, 1e-2), (100, 5e-3), (200, 1e-3), (300, 1e-4)], 
    'snapshot': 50,
    'console_print': 5,    
    'optim_type': 'Adam', 
    'random_seed': 1977,
    'num_workers': 0
}

config['model_opt'] = model_opt
config['algorithm_type'] = 'APM_boxstacking'
