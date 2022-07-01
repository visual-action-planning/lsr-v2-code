import argparse
import os
from importlib.machinery import SourceFileLoader
import algorithms as alg
from dataloader import APNDataset

parser = argparse.ArgumentParser()
parser.add_argument('--exp_apn', type=str, required=True, default='', 
                    help='config file with parameters of the model')
parser.add_argument('--seed', type=int, default=1977, 
                    help='random seed')
parser.add_argument('--chpnt_path', type=str, default='', 
                    help='path to the checkpoint')
parser.add_argument('--num_workers', type=int, default=0,      
                    help='number of data loading workers')
parser.add_argument('--train_apn' , type=int, default=1, 
                    help='trains the apn network')
args_opt = parser.parse_args()


# Load ActionProposalNetwork (APN) config file
apn_exp_name = args_opt.exp_apn + '_seed' + str(args_opt.seed)
apn_config_file = os.path.join('.', 'configs', args_opt.exp_apn + '.py')
apn_directory = os.path.join('.', 'models', apn_exp_name) 
print(apn_directory)
if (not os.path.isdir(apn_directory)):
    os.makedirs(apn_directory)

apn_config = SourceFileLoader(args_opt.exp_apn, apn_config_file).load_module().config 
apn_config['model_opt']['random_seed'] = args_opt.seed
apn_config['model_opt']['exp_name'] = apn_exp_name
apn_config['model_opt']['exp_dir'] = apn_directory # place where logs, models, and other stuff will be stored
print(' *- Loading experiment %s from file: %s' % (args_opt.exp_apn, apn_config_file))
print(' *- Generated logs, snapshots, and model files will be stored on %s' % (apn_directory))

# Initialise VAE model
algorithm = getattr(alg, apn_config['algorithm_type'])(apn_config['model_opt'])
print(' *- Loaded {0}'.format(apn_config['algorithm_type']))

data_train_opt = apn_config['data_train_opt']
train_dataset = APNDataset(
        task_name=data_train_opt['task_name'],
        dataset_name=data_train_opt['dataset_name'], 
        split=data_train_opt['split'],
        random_seed=args_opt.seed,
        dtype=data_train_opt['dtype'], 
        img_size=data_train_opt['img_size'])

data_test_opt = apn_config['data_test_opt']
test_dataset = APNDataset(
        task_name=data_test_opt['task_name'],
        dataset_name=data_test_opt['dataset_name'], 
        split=data_test_opt['split'],
        random_seed=args_opt.seed,    
        dtype=data_test_opt['dtype'],
        img_size=data_test_opt['img_size'])

assert(test_dataset.dataset_name == train_dataset.dataset_name)
assert(train_dataset.split == 'train')
assert(test_dataset.split == 'test')

if args_opt.num_workers is not None:
    num_workers = args_opt.num_workers    
else:
    num_workers = apn_config_file['model_opt']['num_workers']

if args_opt.chpnt_path != '':
    args_opt.chpnt_path = apn_directory + args_opt.chpnt_path
    
if args_opt.train_apn:
    algorithm.train(train_dataset, test_dataset, num_workers, args_opt.chpnt_path)
