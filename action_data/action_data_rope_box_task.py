import action_data.utils_action_data as utils
import pickle
import os, os.path
import torch
from importlib.machinery import SourceFileLoader
import numpy as np
from random import shuffle
import sys
sys.path.append('../architectures/')


def generate_data_with_seed(vae_name, apn_exp_name, split, random_seed):
    """
    vae_name: VAE config file
    apn_exp_name: APN config file
    split: 'test'/'train'
    """
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    print(' *- Random seed set to ', random_seed)
    
    # Load VAE config
    vae_config_file = os.path.join('.', 'configs', vae_name + '.py')
    vae_config = SourceFileLoader(vae_name, vae_config_file).load_module().config 
    print(' *- VAE loaded from: ', vae_config_file)
    
    # Load APN config
    apn_config_file = os.path.join('.', 'configs', apn_exp_name + '.py')
    apn_config = SourceFileLoader(apn_exp_name, apn_config_file).load_module().config 
    print(' *- APN loaded from: ', apn_config_file)
    
    # Load VAE
    opt = vae_config['vae_opt']
    opt['exp_name'] = vae_name
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt['device'] = device
    opt['vae_load_checkpoint'] = False
    opt['n_latent_samples'] = apn_config[f'data_{split}_opt']['n_latent_samples']
    print(' *- # latent spamples ', opt['n_latent_samples'])
    vae = utils.init_vae(opt)
    
    # Load the original data
    data_original_opt_key = 'path_to_original_{0}_data'.format(split)
    print(data_original_opt_key)
    original_data_path = apn_config['data_original_opt'][data_original_opt_key]
    print(original_data_path)
    original_data_path_with_seed = '{0}_seed{1}.pkl'.format(
            original_data_path.split('.pkl')[0], random_seed)
    print(original_data_path_with_seed)
    print(' *- Loading checked and scaled original data from: ', original_data_path_with_seed)
    
    with open(original_data_path_with_seed, 'rb') as f:
        action_data_dict = pickle.load(f)
        threshold_min = action_data_dict['min']
        threshold_max = action_data_dict['max']
        action_data = action_data_dict['data']
        print(' *- Loaded data with thresholds: ', threshold_min, threshold_max, 
              ' and len ', len(action_data))

    # Apn data preparations    
    generated_data_name = 'action_' + vae_name    
    generated_data_dir = 'datasets/action_data/' + generated_data_name
    print(' *- Generated APN data directory: ',generated_data_dir )
    if (not os.path.isdir(generated_data_dir)):
            os.makedirs(generated_data_dir)

    action_data_latent = []
    i = 0
    for img1, img2, coords in action_data:
        i += 1
        
        # VAE forward pass
        img1 = img1.to(device).unsqueeze(0).float()
        img2 = img2.to(device).unsqueeze(0).float()
        
        (enc_mean1, z_samples1, dec_mean_original1, 
         dec_mean_samples1) = utils.vae_forward_pass(img1, vae, opt)
        (enc_mean2, z_samples2, dec_mean_original2, 
         dec_mean_samples2) = utils.vae_forward_pass(img2, vae, opt)
        
        # Save the latent samples and the decodings
        latent_original = [enc_mean1.squeeze().detach(), 
                           enc_mean2.squeeze().detach(), coords]
        action_data_latent.append(latent_original)

        for j in range(opt['n_latent_samples']):
            latent_sample = [z_samples1[j].detach(), z_samples2[j].detach(),
                             coords]
            action_data_latent.append(latent_sample)

    shuffle(action_data_latent)
    print(generated_data_dir)
    print('     - Action_pairs: ', i, '/', len(action_data))
    print('     - Total samples generated: ', len(action_data_latent))
    
    with open(generated_data_dir + '/latent_{0}_seed{1}.pkl'.format(split, random_seed, int(opt['n_latent_samples'])), 
              'wb') as f:
        pickle.dump({'data': action_data_latent, 'min': threshold_min, 
                     'max': threshold_max, 'n_action_pairs': i,
                     'n_original_data': len(action_data),
                     'n_generated_data': len(action_data_latent)}, f)
    
    
def generate_new_spits(dataset_name, path_to_dataset, random_seed):
    """Reads all the data and regenerates the train/test/validation splits."""
    
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    print(' *- Random seed set to ', random_seed)
    
    # like the entire split: rope_2500
    path_to_original_dataset = path_to_dataset + dataset_name + '.pkl'

    with open(path_to_original_dataset, 'rb') as f:
        all_data = pickle.load(f)
        print(' *- Loaded data: ', path_to_original_dataset)
    
    # Rope data has always the thresholds
    threshold_min = np.array([1., 0., 0., 0., 0.]).reshape(-1, 1)
    threshold_max = np.array([2., 2., 2., 2., 2.]).reshape(-1, 1)
    print(' *- Thresholds: ', threshold_min, threshold_max)
    
    scaled_action_data = []
    box_action_pair_counter = 0
    rope_action_pair_counter = 0
    noaction_pair_counter = 0
    # all_data still contains action pairs
    for item in all_data:
        # Filter our action pairs
        if item[2] == 1:
            img1 = torch.tensor(item[0]/255.).float().permute(2, 0, 1)
            img2 = torch.tensor(item[1]/255.).float().permute(2, 0, 1)
            if item[3][0] == 1:
                box_action_pair_counter += 1
                box_action = get_actions_from_classes(item[4], item[5])[1]
                pick_box_x, pick_box_y = box_action[0] // 3, box_action[0] % 3
                place_box_x, place_box_y = box_action[1] // 3, box_action[1] % 3
                action = np.array([item[3][0], pick_box_x, pick_box_y, place_box_x, place_box_y]).reshape(-1, 1)
                action_array_scaled = (action - threshold_min)/(threshold_max - threshold_min) # (3, 1)
                assert(np.all(action_array_scaled) >= 0. and np.all(action_array_scaled) <= 1.)
                scaled_action_data.append([img1, img2, torch.from_numpy(action_array_scaled).float().squeeze()]) # all torch float32
            if item[3][0] == 2:
                rope_action_pair_counter += 1
                action = np.array([item[3][0], -1, -1, -1, -1]).reshape(-1, 1)
                action_array_scaled = (action - threshold_min)/(threshold_max - threshold_min) # (3, 1)
                scaled_action_data.append([img1, img2, torch.from_numpy(action_array_scaled).float().squeeze()]) # all torch float32 
        if item[2] == 0:
            noaction_pair_counter += 1
    print('# action pairs: ', box_action_pair_counter + rope_action_pair_counter)
    print('# rope action pairs: ', rope_action_pair_counter)
    print('# box action pairs: ', box_action_pair_counter)
    print('# no action pairs: ', noaction_pair_counter)
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(scaled_action_data, test_size=0.2, 
                                    shuffle=True, random_state=1201)
    print('     - Len training split: ', len(train))
    print('     - Len test split: ', len(test))
    
    path_to_dataset += 'action_data/'
    if not os.path.exists(path_to_dataset):
        os.makedirs(path_to_dataset)
        
    new_train_data = 'train_action_{0}_seed{1}.pkl'.format(dataset_name, random_seed)
    with open(path_to_dataset + new_train_data, 'wb') as f:
        pickle.dump({'data': train, 'min': threshold_min, 'max': threshold_max}, f)
        print(' *- {0} saved.'.format(path_to_dataset + new_train_data))

    new_test_data = 'test_action_{0}_seed{1}.pkl'.format(dataset_name, random_seed)
    with open(path_to_dataset + new_test_data, 'wb') as f:
        pickle.dump({'data': test, 'min': threshold_min,'max': threshold_max}, f) 
        print(' *- {0} saved.'.format(path_to_dataset + new_test_data))
        
            
def get_actions_from_classes(class1, class2):
    # check if its a rope or a box action
    action_id = -1 # action 1 is box action 2 is rope -1 is no action
    if not (class1[0:9] == class2[0:9]).all():
        action_id = 1
        actions_bi = np.where(class1[0:9] != class2[0:9])
        # check whether the box is in class 1
        if class1[actions_bi[0][0]] > 0:
            actions = [action_id, [actions_bi[0][0], actions_bi[0][1]]]
        else:
            actions = [action_id, [actions_bi[0][1], actions_bi[0][0]]]
        return actions

    if not (class1[9:] == class2[9:]).all():
        action_id = 2
        actions = [action_id, [-1,-1]]
        return actions
    actions = [action_id, [-1,-1]]
    return actions
