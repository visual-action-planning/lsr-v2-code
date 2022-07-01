batch_size = 64

config = {}

# set the parameters related to the training and testing set
data_train_opt = {}
data_train_opt['batch_size'] = batch_size
data_train_opt['epoch_size'] = None
data_train_opt['dataset_name'] = 'box_stacking_normal_task_2500_seed_1122'
data_train_opt['split'] = 'train'
data_train_opt['binarise'] = False
data_train_opt['binarise_threshold'] = None
data_train_opt['img_size'] = 256

data_test_opt = {}
data_test_opt['batch_size'] = batch_size
data_test_opt['unsupervised'] = True
data_test_opt['epoch_size'] = None
data_test_opt['dataset_name'] = 'box_stacking_normal_task_2500_seed_1122'
data_test_opt['split'] = 'test'
data_test_opt['binarise'] = False
data_test_opt['binarise_threshold'] = None
data_test_opt['img_size'] = 256

config['data_train_opt'] = data_train_opt
config['data_test_opt']  = data_test_opt

vae_opt = {
    'model': 'VAE_TinyResNet',
    'filename': 'vae',
    'num_workers': 4,

    'loss_fn': 'fixed decoder variance', 
    'learn_dec_logvar': False,
    'input_dim': 256*256*3,
    'input_channels': 3,
    'latent_dim': 12,
    'out_activation': 'sigmoid',
    'dropout': 0.3,
    'weight_init': 'normal_init',

    'conv1_out_channels': 32,
    'latent_conv1_out_channels': 128,
    'kernel_size': 3,
    'num_scale_blocks': 2,
    'block_per_scale': 1,
    'depth_per_block': 2,
    'fc_dim': 512,
    'image_size': 256,

    'batch_size': batch_size,
    'snapshot': 50,
    'console_print': 1,

    'beta_min': 0,
    'beta_max': 2,
    'beta_max_epoch': 400,
    'beta_steps': 100,
    'kl_anneal': True,

    'gamma_warmup': 0,
    'gamma_min': 100,
    'gamma_max': 100,
    'gamma_steps': 1,
    'gamma_anneal': False,

    'min_dist': 0.0,
    'min_dist_epoch_update': 5,
    'min_dist_step': 0.1,
    'min_dist_percentile': 0,
    'min_dist_reached_nmax': 1,
    'min_distance_update_condition': 'gap_positive',
    'distance_type': '1',

    'epochs': 1000,
    'min_epochs': 500,
    'max_epochs': 1000,
    'lr_schedule': [(0, 1e-03), (20, 1e-04), (300, 1e-5)],
    'optim_type': 'Adam',
    'random_seed': 1122
}

config['vae_opt'] = vae_opt
config['algorithm_type'] = 'MM_algorithm'
