from action_data.action_data_stacking_task import generate_new_spits as generate_new_spits_stacking 
from action_data.action_data_stacking_task import generate_data_with_seed as generate_data_with_seed_stacking

from action_data.action_data_rope_box_task import generate_new_spits as generate_new_spits_rope_box
from action_data.action_data_rope_box_task import generate_data_with_seed as generate_data_with_seed_rope_box


def generate_apm_splits_with_seeds(task):
    print(f'Generating APM splits for task: {task}')
    
    path_to_dataset = 'datasets/'
    seeds = [1977]
    
    if task == 'box_stacking_normal_task':
        dataset = "box_stacking_normal_task_2500"
        for seed in seeds:
            print('Seed {0}'.format(seed))
            generate_new_spits_stacking(dataset, path_to_dataset, seed)
    elif task == 'box_stacking_hard_task':
        dataset = "box_stacking_hard_task_2500"
        for seed in seeds:
            print('Seed {0}'.format(seed))
            generate_new_spits_stacking(dataset, path_to_dataset, seed)
    elif task == 'rope_box_task':
        dataset = "rope_box_task_2500"
        for seed in seeds:
            print('Seed {0}'.format(seed))
            generate_new_spits_rope_box(dataset, path_to_dataset, seed)
    else:
        raise ValueError('Task not recognized.')
    print(f'Done: generating APM splits for task: {task}')


def process_apm_splits_with_mm(task):
    print(f'Processing APM data with MM for task: {task}')
    
    seeds = [1977]
    if task == 'box_stacking_normal_task':
        vae_name = 'box_stacking_normal_task_MM'
        apn_exp_name = 'box_stacking_normal_task_APM'
        for seed in seeds:
            print('Seed {0}'.format(seed))
            print('Seed {0}'.format(seed))
            generate_data_with_seed_stacking(vae_name, apn_exp_name, 'train', seed)
            generate_data_with_seed_stacking(vae_name, apn_exp_name, 'test', seed)
    elif task == 'box_stacking_hard_task':
        vae_name = 'box_stacking_hard_task_MM'
        apn_exp_name = 'box_stacking_hard_task_APM'
        for seed in seeds:
            print('Seed {0}'.format(seed))
            print('Seed {0}'.format(seed))
            generate_data_with_seed_stacking(vae_name, apn_exp_name, 'train', seed)
            generate_data_with_seed_stacking(vae_name, apn_exp_name, 'test', seed)
    elif task == 'rope_box_task':
        vae_name = 'rope_box_task_MM'
        apn_exp_name = 'rope_box_task_APM'
        for seed in seeds:
            print('Seed {0}'.format(seed))
            print('Seed {0}'.format(seed))
            generate_data_with_seed_rope_box(vae_name, apn_exp_name, 'train', seed)
            generate_data_with_seed_rope_box(vae_name, apn_exp_name, 'test', seed)
    else:
        raise ValueError('Task not recognized.')
    
    print(f'Done: processing APM data with MM for task {task}')

def main():
    tasks = ['box_stacking_normal_task', 
             'box_stacking_hard_task', 
             'rope_box_task']
    for task in tasks:
        generate_apm_splits_with_seeds(task)
        process_apm_splits_with_mm(task)

if __name__== "__main__":
    main()